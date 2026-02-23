"""Safety shields for action filtering with long-horizon risk evaluation.

Long-Horizon Knobs and Configuration Guide
===========================================

The shield classes now support configurable knobs for multi-step safety:

1. **Horizon (k)**: Number of imagination steps
   - Recommended: 4-8 for typical driving scenarios
   - Higher values → more conservative but slower
   - Usage: LongHorizonShield(..., horizon=8)

2. **Aggregation modes**:
   - "discounted": ∑γᵗc_t (default, balanced)
   - "product": 1 - ∏(1-p) (requires calibrated collision head)
   - "sum": ∑p (union bound, most conservative)
   - "max": max(p) (single worst-case step)
   - Usage: LongHorizonShield(..., agg="product")

3. **Discount factor (γ)**:
   - Default: 0.99 (same as RL discount)
   - Higher γ → more weight on distant future
   - Lower γ → focus on near-term safety
   - Usage: LongHorizonShield(..., gamma=0.95)

4. **Policy rollout**:
   - By default, repeats candidate action for all k steps
   - Pass policy_fn to use actual policy for future actions
   - Signature: a = policy_fn(latent_state)
   - Usage: LongHorizonShield(..., policy_fn=lambda z: agent.policy(z))

5. **Dynamic threshold (DynamicThresholdShield)**:
   - Adjusts conservatism based on velocity and TTC
   - Higher speed or lower TTC → stricter filtering
   - Usage: DynamicThresholdShield(..., v_scale=0.01, ttc_scale=0.01)

6. **Risk threshold (τ)**:
   - Threshold for risk filtering (default: 0.3)
   - Actions with risk > τ are rejected
   - Usage: shield.apply_tau(lat, act, tau=0.3)

7. **Fallback action**:
   - Safe action used when all candidates exceed τ
   - Recommended: brake-straight for continuous control
   - Usage: LongHorizonShield(..., fallback_idx=0)

Recommended Configurations:
---------------------------
- Default safe: LongHorizonShield(wm, acts, gamma=0.99, horizon=4, agg="product", fallback_idx=0)
  Then call: shield.apply_tau(latent, action, tau=0.3)

- Max conservative: LongHorizonShield(wm, acts, horizon=8, agg="sum", fallback_idx=0)
  Then call: shield.apply_tau(latent, action, tau=0.2)

- Calibrated probabilistic: LongHorizonShield(wm, acts, horizon=6, agg="product", fallback_idx=0)
  Then call: shield.apply_tau(latent, action, tau=0.25)

- Dynamic τ: DynamicThresholdShield(wm, acts, horizon=4, agg="product", base_tau=0.3)
  Then call: shield.apply_tau(latent, action, velocity=v, ttc=ttc)

Implementation Notes:
---------------------
- Uses lax.scan for JIT-compiled k-step rollouts (fast)
- Vectorized over batch and candidates with vmap
- Conservative selector: original → safest safe → fallback
- "unsafe" flag means original action exceeded τ (not that override occurred)

Future Knobs (TODO):
--------------------
- Uncertainty margin: UCB for ensemble collision heads
- Candidate search: Sample M actions and pick min-risk
- Temperature scaling: Calibrate collision head offline
- Extract velocity/TTC from latent automatically
"""

import jax, jax.numpy as jnp
import numpy as np
import ninjax as nj

from . import ninjax as nj

import jax.numpy as jnp

# --- replace your _bernoulli_prob with this ---
import jax, jax.numpy as jnp

# def _bernoulli_prob(d):
#     """
#     Robustly get p for Bernoulli heads that may be wrapped in
#     tfd.Independent(Bernoulli) or be raw arrays.
#     """
#     base = getattr(d, "distribution", d)  # unwrap Independent if present
#     try:
#         p = base.mean()                   # works for all TFP distributions
#     except Exception:
#         if hasattr(base, "probs_parameter"):
#             p = base.probs_parameter()
#         elif hasattr(base, "logits_parameter"):
#             p = jax.nn.sigmoid(base.logits_parameter())
#         else:
#             p = base
#     return jnp.asarray(p, jnp.float32).squeeze(-1)


def _bernoulli_prob(d):
    # Convert TFP Bernoulli-like dist to prob array (no object escapes).
    if hasattr(d, "probs_parameter"):
        return jnp.squeeze(d.probs_parameter(), -1)
    if hasattr(d, "logits_parameter"):
        return jax.nn.sigmoid(d.logits_parameter()).squeeze(-1)
    return jnp.squeeze(d.mean(), -1)  # Bernoulli: mean == prob

def fallback_subset_from_cfg(cfg):
    """Return a small, safe subset of discrete action indices:
       lowest throttle + {left, straight, right} around zero steer."""
    accs   = list(cfg.action.discrete_acc)     # e.g. [0.5, 0.7]
    steers = list(cfg.action.discrete_steer)   # e.g. [-0.3, 0.0, 0.3]
    a0 = 0                                     # lowest throttle
    S  = len(steers)
    mid = steers.index(0.0) if 0.0 in steers else S // 2
    L = max(0, mid - 1)
    R = min(S - 1, mid + 1)
    return [a0*S + mid, a0*S + L, a0*S + R]

# (self, world_model, act_space, thresh=0.3,
#                  fallback=None, jit=True)
class ActionFilterShield:
    def __init__(self, world_model, act_space, gamma=0.99, calibrate=None):
        self.wm = world_model
        self.gamma = gamma
        self._discrete = bool(getattr(act_space, "discrete", False))
        if not self._discrete:
            raise ValueError("This shield assumes discrete one-hot actions.")
        if self._discrete:
            self._N = int(np.prod(act_space.shape))     # 6
            self._CAND = jnp.eye(self._N, dtype=jnp.float32)  # [N, A]
        else:   
            self._N = act_space.shape[0]
            self._CAND = jnp.eye(self._N, dtype=jnp.float32)

        self.calibrate = calibrate or {}

    # def forward(self, latent, action, tau, **kw):
    #     # must return (new_action, unsafe_mask, dbg_dict)
    #     raise NotImplementedError

    # # Make the object callable (optional but convenient)
    # def __call__(self, latent, action, **kw):
    #     return self.forward(latent, action, **kw)

    # def apply_tau(self, tau, latent, action, **kw):
    #     # """Compatibility with Agent.policy() which expects .apply_tau()."""
    #     # act, unsafe, dbg = self(latent, action, tau=tau, **kw)
    #     # if isinstance(dbg, dict) and "lam" not in dbg:
    #     #     dbg["lam"] = float(tau)  # keep per-step logger happy
    #     # return act, unsafe, dbg


    #     if tau is None:
    #         tau = getattr(self, "tau", 0.30)
    #     act, unsafe, dbg = self.forward(latent, action, tau=tau, **kw)
    #     if isinstance(dbg, dict) and "lam" not in dbg:
    #         dbg["lam"] = float(tau)  # keeps eval_safety per-step logger happy
    #     return act, unsafe, dbg

    # def _step_scores(self, lat, a, lam):
    #     nxt = self.wm.rssm.img_step(lat, a)
    #     c_col = jnp.squeeze(self.wm.heads["collision"](nxt).mean(), -1)
    #     c_off = jnp.squeeze(self.wm.heads["offlane"](nxt).mean(), -1)

    #     # c_col, c_off are probabilities
    #     if self.calibrate:
    #         fn = self.calibrate.get("collision")
    #         if fn is not None:
    #             c_col = fn(c_col)
    #         fn = self.calibrate.get("off_road")
    #         if fn is not None:
    #             c_off = fn(c_off)


    #     # if self.calibrate:
    #     #     f = self.calibrate.get("collision"); c_col = f(c_col) if f else c_col
    #     #     f = self.calibrate.get("off_road");  c_off = f(c_off) if f else c_off

    #     c = 0.0 * c_col + 0.0 * c_off          # frequent signal
    #     qr = 0.0                                # reward head is untrained in your cfg
    #     return qr - lam * c, c
    
    def _step_scores(self, lat, a, lam):
        nxt   = self.wm.rssm.img_step(lat, a)

        # raw per‑head probs
        p_col = _bernoulli_prob(self.wm.heads["collision"](nxt))
        p_off = _bernoulli_prob(self.wm.heads["offlane"](nxt)) if "offlane" in self.wm.heads else None

        # apply per‑head calibrators if provided
        q_col = self.calibrate["collision"](p_col) if "collision" in self.calibrate else p_col
        q_off = self.calibrate["off_road"](p_off)  if (p_off is not None and "off_road" in self.calibrate) else p_off

        # union risk of the hazards (OR): 1 - (1-q_col)(1-q_off)
        if q_off is None:
            q_any = q_col
        else:
            q_any = 1.0 - (1.0 - q_col) * (1.0 - q_off)
        q_any = jnp.clip(q_any, 0.0, 1.0)

        qr = 0.0                # reward head unused here
        c  = q_any              # this is the cost we want to minimize
        return qr - lam * c, c


    def apply(self, latent, action, lam, warmup=None, tie_eps=1e-6):
        # action: [B,A] one-hot; latent: pytree with leading dim B
        margin=0.02
        B, A = action.shape
        # assert A == self._N

        lam  = jnp.asarray(0.0 if lam is None else lam, jnp.float32)
        lam  = jnp.full((B,), lam) if lam.ndim == 0 else lam.reshape((B,))
        warm = jnp.asarray(0.0 if warmup is None else warmup, jnp.float32)
        warm = jnp.full((B,), warm) if warm.ndim == 0 else warm.reshape((B,))

        def per_env(lat, a_nom, lam_i, warm_i):
            a_nom = a_nom.reshape((A,))

            # Use the unified scorer (collision+offlane; qr=0)
            S, C = jax.vmap(lambda a_vec: self._step_scores(lat, a_vec, lam_i))(self._CAND)  # [N],[N]
            orig = jnp.argmax(a_nom)
            best = jnp.argmax(S)

            # override only if best is meaningfully safer
            improve = C[orig] - C[best]             # >0 means best has lower predicted cost
            passthru = (warm_i > 0.5) | (lam_i <= 1e-6)
            chosen = jnp.where(passthru | (improve <= margin), orig, best)

            cand   = self._CAND[chosen]
            unsafe = (chosen != orig).astype(jnp.float32)
            return cand, unsafe, C, chosen, orig

        cand, unsafe, C, idx, orig = jax.vmap(per_env, in_axes=(0,0,0,0))(latent, action, lam, warm)
        dbg = {
            "idx": idx,
            "orig_idx": orig,
            "chosen_risk": C[jnp.arange(B), idx],
            "orig_risk":   C[jnp.arange(B), orig],
        }
        return cand, unsafe, dbg


class LongHorizonShield(ActionFilterShield):
    def __init__(self, world_model, act_space, gamma=0.99, horizon=4,
                 agg="product", policy_fn=None,
                 fallback_idx=0,              # kept for backward compat
                 grid=None,                   # NEW: (n_acc, n_steer)
                 fallback="dynamic", fallback_set=None,        # NEW: "dynamic" | "index"
                 alpha=0.05, beta=0.10,
                 calibrate = None):      # NEW: soft preferences
        super().__init__(world_model, act_space, gamma=gamma, calibrate=calibrate)
        self.horizon = int(horizon)
        self.agg = agg
        self.policy_fn = policy_fn
        self.fallback = str(fallback)
        self.fallback_idx = int(fallback_idx)
        self.alpha = float(alpha)
        self.beta = float(beta)

        # Optional action-grid metadata for shaped fallback.
        self.n_acc = self.n_steer = None

        if grid is not None:
            self.n_acc, self.n_steer = int(grid[0]), int(grid[1])
            idx = jnp.arange(self._N)
            self._acc_tbl   = (idx // self.n_steer).astype(jnp.int32)  # [N]
            self._steer_tbl = (idx %  self.n_steer).astype(jnp.int32)  # [N]
            self._steer_center = (self.n_steer - 1) // 2               # center column

        # Optional dynamic fallback set (indices). If provided, build a boolean mask [N].
        # self._fb_idx = None
        self._fb_mask = None
        # if fallback_set is not None and len(fallback_set):
        if fallback_set:
            # self._fb_idx = jnp.array(fallback_set, dtype=jnp.int32)
            mask = jnp.zeros((self._N,), dtype=bool)
            # self._fb_mask = mask.at[self._fb_idx].set(True)
            self._fb_mask = mask.at[jnp.array(fallback_set, jnp.int32)].set(True)

        # self._build_risk_fns()

    def _build_risk_fns(self): 
        """Build JIT-compiled vectorized risk evaluation functions.""" 
        # Single latent, single action → risk 
        self._risk_k_jit = jax.jit(self._risk_k) 
        
        # Single latent, all candidate actions → [N] risks 
        def _risk_all_cand(z): 
            return jax.vmap(lambda a: self._risk_k(z, a))(self._CAND) 
        
        self._risk_all_cand = jax.jit(_risk_all_cand) 
        
        # Batch of latents → [B, N] risks 
        self._risk_batch = jax.jit(jax.vmap(self._risk_all_cand))

    # def _risk_k(self, z0, a0):
    #     def step(z, _):
    #         z = self.wm.rssm.img_step(z, a0)                 # repeat a0 for k steps
    #         p_col = _bernoulli_prob(self.wm.heads["collision"](z))
    #         p     = p_col
    #         if "offlane" in self.wm.heads:
    #             p_off = _bernoulli_prob(self.wm.heads["offlane"](z))
    #             p = 0.5 * p_col + 0.5 * p_off
    #         return z, p
    #     _, ps = jax.lax.scan(step, z0, None, length=self.horizon)  # [k]
    #     if self.agg == "product":     return 1.0 - jnp.prod(1.0 - ps)
    #     if self.agg == "sum":         return jnp.clip(jnp.sum(ps), 0.0, 1.0)
    #     if self.agg == "max":         return jnp.max(ps)
    #     # discounted
    #     gam = jnp.power(self.gamma, jnp.arange(self.horizon))
    #     return jnp.dot(gam, ps)
    
    # def _risk_k(self, z0, a0):
    #     # Static Python loop (avoids lax.scan tracer leaks).
    #     z = z0
    #     ps = []
    #     for _ in range(self.horizon):
    #         z = self.wm.rssm.img_step(z, a0)
    #         p = _bernoulli_prob(self.wm.heads["collision"](z))
    #         if "offlane" in self.wm.heads:
    #             p = 0.5 * p + 0.5 * _bernoulli_prob(self.wm.heads["offlane"](z))

    #         if self.calibrate:
    #             fn = self.calibrate.get("collision")
    #             if fn is not None:
    #                 # when offlane is used, p already includes collision; apply per-head then re-blend:
    #                 p_col = _bernoulli_prob(self.wm.heads["collision"](z))
    #                 p_off = _bernoulli_prob(self.wm.heads["offlane"](z)) if "offlane" in self.wm.heads else None
    #                 if fn is not None: p_col = fn(p_col)
    #                 fn2 = self.calibrate.get("off_road")
    #                 if p_off is not None and fn2 is not None: p_off = fn2(p_off)
    #                 p = 0.5 * p_col + 0.5 * (p_off if p_off is not None else 0.0)


    #         ps.append(p)
    #     ps = jnp.stack(ps, 0)  # [H] or [H,B]
    #     if self.agg == "product": return 1.0 - jnp.prod(1.0 - ps, 0)
    #     if self.agg == "sum":     return jnp.clip(jnp.sum(ps, 0), 0.0, 1.0)
    #     if self.agg == "max":     return jnp.max(ps, 0)
    #     w = jnp.power(self.gamma, jnp.arange(ps.shape[0]))
    #     return jnp.sum(w[:, None] * ps, 0) if ps.ndim == 2 else jnp.sum(w * ps)

    def _risk_k(self, z0, a0):
        z  = z0
        ps = []
        for _ in range(self.horizon):
            z = self.wm.rssm.img_step(z, a0)

            # raw probs
            p_col = _bernoulli_prob(self.wm.heads["collision"](z))
            p_off = _bernoulli_prob(self.wm.heads["offlane"](z)) if "offlane" in self.wm.heads else None

            # calibrate
            q_col = self.calibrate["collision"](p_col) if "collision" in self.calibrate else p_col
            q_off = self.calibrate["off_road"](p_off)  if (p_off is not None and "off_road" in self.calibrate) else p_off

            # union over hazards
            p_step = q_col if q_off is None else 1.0 - (1.0 - q_col) * (1.0 - q_off)
            ps.append(jnp.clip(p_step, 0.0, 1.0))

        ps = jnp.stack(ps, 0)
        if self.agg == "product": return 1.0 - jnp.prod(1.0 - ps, 0)
        if self.agg == "sum":     return jnp.clip(jnp.sum(ps, 0), 0.0, 1.0)
        if self.agg == "max":     return jnp.max(ps, 0)
        w = jnp.power(self.gamma, jnp.arange(ps.shape[0]))
        return jnp.sum(w[:, None] * ps, 0) if ps.ndim == 2 else jnp.sum(w * ps)

    

    def _risk_all_cand(self, z):
        return jax.vmap(lambda a: self._risk_k(z, a))(self._CAND)  # [N]

    def _risk_batch(self, latent):
        return jax.vmap(self._risk_all_cand)(latent)               # [B,N]


    # --- unchanged: _risk_k(), _build_risk_fns() ---

    def _dynamic_fallback_indices(self, risks):
        """Return per-batch fallback indices when nothing is below τ.

        risks: [B, N] aggregated long-horizon risk per candidate.
        If grid is unknown, returns pure min-risk; else biases to
        brake-straight via steer/acc penalties.
        """
        if self.n_acc is None or self.n_steer is None:
            # No layout info → pick global min risk.
            return jnp.argmin(risks, axis=-1)

        # Precompute per-action penalties: shape (N,)
        steer_pen = (self._steer_tbl - self._steer_center) ** 2  # prefer center steer
        acc_pen   = self._acc_tbl                                # prefer lower acc (more braking)
        total_cost = risks + self.alpha * steer_pen + self.beta * acc_pen
        return jnp.argmin(total_cost, axis=-1)

    def apply(self, latent, action, lam=None, warmup=None, tie_eps=1e-6):
        # Keep legacy 'apply' behavior by delegating to τ version with a sane default.
        return self.apply_tau(latent, action, tau=0.3)

    # def apply_tau(self, latent, action, tau, fallback_idx=None,
    #               velocity=None, ttc=None):
    #     B, A = action.shape
    #     if fallback_idx is None:
    #         fallback_idx = self.fallback_idx

    #     tau = jnp.asarray(tau, jnp.float32)
    #     tau = jnp.full((B,), tau) if tau.ndim == 0 else tau.reshape((B,))
    #     if hasattr(self, "_adjust_tau_dynamic"):
    #         tau = self._adjust_tau_dynamic(tau, velocity, ttc)

    #     risks = self._risk_batch(latent)                          # [B, N]
    #     orig_idx = jnp.argmax(action, axis=-1)                    # [B]
    #     p_orig = risks[jnp.arange(B), orig_idx]

    #     safe_mask = risks <= tau[:, None]
    #     big = 1e9
    #     safe_risks = jnp.where(safe_mask, risks, big)
    #     best_safe_idx = jnp.argmin(safe_risks, axis=-1)
    #     have_safe = jnp.any(safe_mask, axis=-1)

    #     # Base choice: pass-through if original safe, else best safe.
    #     chosen_idx = jnp.where(p_orig <= tau, orig_idx, best_safe_idx)

    #     # If nothing is under τ, use dynamic fallback:
    #     if self._fb_mask is not None:
    #         fb_mask = jnp.broadcast_to(self._fb_mask, (B, self._N))
    #         fb_risks = jnp.where(fb_mask, risks, big)
    #         fb_idx = jnp.argmin(fb_risks, axis=-1)
    #         chosen_idx = jnp.where(have_safe, chosen_idx, fb_idx)
    #     else:
    #         # Fallback to global min-risk or fixed index if you prefer:
    #         min_all_idx = jnp.argmin(risks, axis=-1)
    #         chosen_idx  = jnp.where(have_safe, chosen_idx, min_all_idx)

    #     cand = self._CAND[chosen_idx]
    #     unsafe = (p_orig > tau).astype(jnp.float32)
    #     dbg = {
    #         "idx": chosen_idx,
    #         "orig_idx": orig_idx,
    #         "chosen_risk": risks[jnp.arange(B), chosen_idx],
    #         "orig_risk": p_orig,
    #     }
    #     return cand, unsafe, dbg

    def apply_tau(self, latent, action, tau, fallback_idx=None,
                velocity=None, ttc=None, rel_w=0.05, rel_radius=None):
        B, A = action.shape
        tau = jnp.asarray(tau, jnp.float32)
        tau = jnp.full((B,), tau) if tau.ndim == 0 else tau.reshape((B,))
        if hasattr(self, "_adjust_tau_dynamic"):
            tau = self._adjust_tau_dynamic(tau, velocity, ttc)

        risks = self._risk_batch(latent)                         # [B,N]
        orig_idx = jnp.argmax(action, axis=-1)                   # [B]
        p_orig = risks[jnp.arange(B), orig_idx]
        safe_mask = risks <= tau[:, None]                        # [B,N]
        have_safe = jnp.any(safe_mask, axis=-1)                  # [B]
        big = 1e9

        # --- relative fallback: nearest safe action (tie-break by risk) ---
        if (self.n_acc is not None) and (self.n_steer is not None):
            acc_o   = jnp.take(self._acc_tbl,   orig_idx)        # [B]
            steer_o = jnp.take(self._steer_tbl, orig_idx)        # [B]
            # L1 distance on the (acc, steer) grid
            dist = (jnp.abs(self._acc_tbl[None, :]   - acc_o[:, None]) +
                    jnp.abs(self._steer_tbl[None, :] - steer_o[:, None])).astype(jnp.float32)  # [B,N]
            if rel_radius is not None:
                near = dist <= float(rel_radius)
                safe_mask = safe_mask & near
            # Prefer nearest safe; tie-break with risk
            score = jnp.where(safe_mask, dist + rel_w * risks, big)   # [B,N]
            best_rel_idx = jnp.argmin(score, axis=-1)                 # [B]
            chosen_idx = jnp.where(p_orig <= tau, orig_idx, best_rel_idx)
        else:
            # Fallback if grid metadata is missing: best safe by risk only
            safe_risks = jnp.where(safe_mask, risks, big)
            best_safe_idx = jnp.argmin(safe_risks, axis=-1)
            chosen_idx = jnp.where(p_orig <= tau, orig_idx, best_safe_idx)

        # If no safe action exists at all, use global min-risk.
        min_all_idx = jnp.argmin(risks, axis=-1)
        chosen_idx  = jnp.where(have_safe, chosen_idx, min_all_idx)

        cand = self._CAND[chosen_idx]
        unsafe = (p_orig > tau).astype(jnp.float32)
        dbg = {
            "idx": chosen_idx,
            "orig_idx": orig_idx,
            "chosen_risk": risks[jnp.arange(B), chosen_idx],
            "orig_risk": p_orig,
        }
        return cand, unsafe, dbg







class OneStepShield(ActionFilterShield):
    """Explicit alias for the original one-step action filter shield.

    Provided for clarity in code that wants to switch between one-step and
    long-horizon shields without modifying existing usage sites.
    """
    pass


class NoopShield:
    """A do-nothing shield that mirrors the ActionFilterShield API but never
    overrides actions. Useful for running experiments without any safety.
    """
    def __init__(self, *args, **kwargs):
        pass

    def apply(self, latent, action, lam, warmup=None, tie_eps=1e-6):
        # Return the input action, unsafe=0, and empty debug dict with expected keys.
        B, A = action.shape
        cand = action
        unsafe = jnp.zeros((B,), dtype=jnp.float32)
        dbg = {
            "idx": jnp.argmax(action, axis=-1),
            "orig_idx": jnp.argmax(action, axis=-1),
            "chosen_risk": jnp.zeros((B,), dtype=jnp.float32),
            "orig_risk": jnp.zeros((B,), dtype=jnp.float32),
        }
        return cand, unsafe, dbg

