import jax
import jax.numpy as jnp
import os
import numpy as np
from dreamerv3.calibration import TemperatureScaler


tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

from . import behaviors, jaxagent, jaxutils, nets
from . import ninjax as nj

from dreamerv3 import safety as safety_mod

from dreamerv3.safety import fallback_subset_from_cfg

from dreamerv3.calibration import CalibManager


# Shield classes are created dynamically below based on config.safe_train


@jaxagent.Wrapper
class Agent(nj.Module):
    def __init__(self, obs_space, act_space, step, config):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step
        self.wm = WorldModel(obs_space, act_space, config, name="wm")
        self.task_behavior = getattr(behaviors, config.task_behavior)(self.wm, self.act_space, self.config, name="task_behavior")
        if config.expl_behavior == "None":
            self.expl_behavior = self.task_behavior
        else:
            self.expl_behavior = getattr(behaviors, config.expl_behavior)(self.wm, self.act_space, self.config, name="expl_behavior")

        

        # self.calib = CalibManager(
        #     hazards=("collision","off_road"),
        #     n_bins=15, min_total=5000, min_pos=200, update_every=5000
        #     )

        # Create shield according to configuration. If safe_train.enable is
        # False, use a NoopShield so actions are not modified (important for
        # intrinsic-only pretraining). If enabled, choose between one-step and
        # long-horizon shields based on config.safe_train.mode.
        # from dreamerv3 import safety as safety_mod
        # safe_cfg = getattr(config, "safe_train", {}) or {}
        # if safe_cfg.get("enable", False):
        #     mode = safe_cfg.get("mode", "one_step")
        #     gamma = safe_cfg.get("gamma", 0.99)
        #     if mode == "long":
        #         horizon = int(safe_cfg.get("horizon", 15))
        #         self.shield = safety_mod.LongHorizonShield(self.wm, act_space["action"], gamma=gamma, horizon=horizon)
        #     else:
        #         self.shield = safety_mod.ActionFilterShield(self.wm, act_space["action"], gamma=gamma)
        # else:
        #     self.shield = safety_mod.NoopShield()

        # at top of agent module
        import numpy as np
        import jax.numpy as jnp
        import jax

        def _logit(x):
            x = jnp.clip(x, 1e-6, 1-1e-6)
            return jnp.log(x) - jnp.log1p(-x)

        def _make_temp_scaler(T, b):
            s = TemperatureScaler()
            s.T = jnp.array(float(T), jnp.float32)
            s.b = jnp.array(float(b), jnp.float32)
            s.fitted = True
            return s.apply  # return callable(p) -> calibrated p

        def load_calibrators_npz(path):
            if not path or not os.path.exists(path):
                return {}
            arr = np.load(path, allow_pickle=True)
            out = {}

            # Case A: nested objects: arr["collision"] -> {"T": ..., "b": ...}
            for key in arr.files:
                obj = arr[key]
                if isinstance(obj, np.ndarray) and obj.dtype == object:
                    try:
                        d = obj.item()
                    except Exception:
                        d = None
                    if isinstance(d, dict) and "T" in d and "b" in d:
                        out[key] = _make_temp_scaler(d["T"], d["b"])

            if not out:
                # Case B: flat keys: collision_T, collision_b, off_road_T, off_road_b
                hazards = set(k.rsplit("_", 1)[0] for k in arr.files if k.endswith(("_T", "_b")))
                for h in hazards:
                    Tk, bk = f"{h}_T", f"{h}_b"
                    if Tk in arr.files and bk in arr.files:
                        T = arr[Tk].item() if hasattr(arr[Tk], "item") else float(arr[Tk])
                        b = arr[bk].item() if hasattr(arr[bk], "item") else float(arr[bk])
                        out[h] = _make_temp_scaler(T, b)

            # aliases (your heads are named "offlane")
            if "off_road" in out and "offlane" not in out:
                out["offlane"] = out["off_road"]
            if "offlane" in out and "off_road" not in out:
                out["off_road"] = out["offlane"]
            return out



        from dreamerv3 import safety as safety_mod
        # ---- training-time shield (optional) ----
        trn_cfg = getattr(config, "safe_train", {}) or {}
        if trn_cfg.get("enable", False):
            mode  = trn_cfg.get("mode", "one_step")
            gamma = float(trn_cfg.get("gamma", 0.99))
            if mode == "long":
                horizon = int(trn_cfg.get("horizon", 6))
                self.shield = safety_mod.LongHorizonShield(
                    self.wm, act_space["action"], gamma=gamma, horizon=horizon
                )
            else:
                self.shield = safety_mod.ActionFilterShield(self.wm, act_space["action"], gamma=gamma)
        else:
            self.shield = safety_mod.NoopShield()

        # cal_map = None
        # if self._calib_params:
        #     cal_map = {
        #         "collision": lambda p: self._cal_apply(self._calib_params["collision"], p),
        #         "off_road":  lambda p: self._cal_apply(self._calib_params["off_road"],  p),
        #     }


        # ---- evaluation-time shield (independent of training) ----
        ev_cfg = getattr(config, "safe_eval", {}) or {}
        calibrate = load_calibrators_npz(ev_cfg.get("calibrator_path", ""))

        if ev_cfg.get("enable", False):
            mode    = ev_cfg.get("mode", "one_step")
            gamma   = float(ev_cfg.get("gamma", 0.99))
            agg     = ev_cfg.get("agg", "product")
            horizon = int(ev_cfg.get("horizon", 6))

            # grid: accept dict {n_acc, n_steer} or tuple/list
            grid_raw = ev_cfg.get("grid", None)
            grid = None
            if grid_raw is not None:
                if isinstance(grid_raw, dict):
                    grid = (int(grid_raw.get("n_acc")), int(grid_raw.get("n_steer")))
                else:
                    g = tuple(grid_raw); grid = (int(g[0]), int(g[1]))

            # dynamic fallback candidate subset from cfg.action.*
            try:
                fb_set = fallback_subset_from_cfg(config)
            except Exception:
                fb_set = None

            if mode == "long":
                self.eval_shield = safety_mod.LongHorizonShield(
                    self.wm, act_space["action"],
                    gamma=gamma, horizon=horizon, agg=agg,
                    grid=grid,
                    fallback=str(ev_cfg.get("fallback", "dynamic")),
                    alpha=float(ev_cfg.get("alpha", 0.05)),
                    beta=float(ev_cfg.get("beta", 0.15)),
                    fallback_idx=int(ev_cfg.get("fallback_idx", 0)),
                    fallback_set=fb_set,
                    calibrate=calibrate
                )
            else:
                self.eval_shield = safety_mod.ActionFilterShield(self.wm, act_space["action"], gamma=gamma, calibrate=calibrate)
        else:
            self.eval_shield = safety_mod.NoopShield()

        # ---- Load calibrator params for EVAL only ----
        self._calib_params = {}
        cal_path = getattr(config, "safe_eval", {}).get("calibrator_path", None)
        if cal_path and os.path.exists(cal_path):
            arr = np.load(cal_path)
            def _get(k):
                return jnp.array(arr[k]) if k in arr.files else None
            self._calib_params = {
                "collision": {"T": _get("collision_T"), "b": _get("collision_b")},
                "off_road":  {"T": _get("off_road_T") or _get("off_T"),
                            "b": _get("off_road_b") or _get("off_b")},
            }



    def policy_initial(self, batch_size):
        return (
            self.wm.initial(batch_size),
            self.task_behavior.initial(batch_size),
            self.expl_behavior.initial(batch_size),
        )

    def train_initial(self, batch_size):
        return self.wm.initial(batch_size)
    
    def _cal_apply(self, params, p):
        if not params or params["T"] is None or params["b"] is None:
            return p
        p = jnp.clip(p, 1e-6, 1 - 1e-6)
        z = (jnp.log(p) - jnp.log1p(-p) - params["b"]) / jnp.maximum(params["T"], 1e-2)
        return jax.nn.sigmoid(z)


    def policy(self, obs, state, mode="train"):
        # print("Mode:", mode)
        self.config.jax.jit and print("Tracing policy function.")
        obs = self.preprocess(obs)
        (prev_latent, prev_action), task_state, expl_state = state
        embed = self.wm.encoder(obs)
        latent, _ = self.wm.rssm.obs_step(prev_latent, prev_action, embed, obs["is_first"])
        self.expl_behavior.policy(latent, expl_state)
        task_outs, task_state = self.task_behavior.policy(latent, task_state)
        expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)

        lam  = obs.get("log_shield_lam",    jnp.zeros_like(obs["is_first"], jnp.float32)).astype(jnp.float32)
        warm = obs.get("log_shield_warmup", jnp.zeros_like(lam,             jnp.float32)).astype(jnp.float32)
   
        # ----- add this small helper inside policy() -----
        # def _attach_risk_logs(outs, latent):
        #     p_col = jnp.squeeze(self.wm.heads["collision"](latent).mean(), -1)  # [B]
        #     p_off = jnp.squeeze(self.wm.heads["offlane"](latent).mean(), -1)    # [B]
        #     outs["log_p_collision"] = p_col
        #     outs["log_p_offlane"]   = p_off
        #     outs["log_p_risk"]      = 0.5 * p_col + 0.5 * p_off
        #     return outs

        # def _attach_risk_logs(outs, latent):
        #     # col_dist = self.wm.heads["collision"](latent)
        #     # off_dist = self.wm.heads["offlane"](latent)
        #     # p_col = getattr(col_dist, "probs", col_dist.mean())  # [B,1] or [B]
        #     # p_off = getattr(off_dist, "probs", off_dist.mean())

        #     p_col = self.wm.heads["collision"](latent).probs.squeeze(-1)
        #     p_off = self.wm.heads["offlane"](latent).probs.squeeze(-1)


        #     # Apply calibration ONLY in eval (presence of self._calib_params is the switch)
        #     if mode == "eval" and self._calib_params:
        #         p_col = self._cal_apply(self._calib_params.get("collision"), p_col)
        #         p_off = self._cal_apply(self._calib_params.get("off_road"),  p_off)

        #     outs["log_p_collision"] = jnp.squeeze(p_col, -1)
        #     outs["log_p_offlane"]   = jnp.squeeze(p_off, -1)
        #     outs["log_p_risk"]      = 0.5 * outs["log_p_collision"] + 0.5 * outs["log_p_offlane"]
        #     return outs
        
        # dreamerv3/agent.py  (inside Agent.policy, helper _attach_risk_logs)
        def _attach_risk_logs(outs, latent):
            def _get_prob(dist):
                base = getattr(dist, "distribution", dist)
                if hasattr(base, "probs_parameter"):
                    p = base.probs_parameter()
                elif hasattr(base, "logits_parameter"):
                    p = jax.nn.sigmoid(base.logits_parameter())
                else:
                    p = dist.mean()  # Independent(Bernoulli).mean() == probability
                return jnp.squeeze(p, -1).astype(jnp.float32)

            p_col = _get_prob(self.wm.heads["collision"](latent))
            p_off = _get_prob(self.wm.heads["offlane"](latent))

            # (optional) if you want a single union risk log:
            outs["log_p_collision"] = p_col
            outs["log_p_offlane"]   = p_off
            outs["log_p_risk"]      = 1.0 - (1.0 - p_col) * (1.0 - p_off)
            return outs


        
        # def _attach_risk_logs(outs, latent):
        #     col_dist = self.wm.heads["collision"](latent)
        #     off_dist = self.wm.heads["offlane"](latent)

        #     # Prefer probabilities; fall back to mean() if .probs not available.
        #     p_col = getattr(col_dist, "probs", col_dist.mean())
        #     p_off = getattr(off_dist, "probs", off_dist.mean())

        #     outs["log_p_collision"] = jnp.squeeze(p_col, -1)
        #     outs["log_p_offlane"]   = jnp.squeeze(p_off, -1)
        #     outs["log_p_risk"]      = 0.5 * outs["log_p_collision"] + 0.5 * outs["log_p_offlane"]
        #     return outs
        # -------------------------------------------------

        # if mode == "eval":
        #     outs = task_outs
        #     outs["action"] = outs["action"].sample(seed=nj.rng())
        #     outs["log_entropy"] = jnp.zeros(outs["action"].shape[:1])

        # if mode == "eval":
        #     outs = task_outs
        #     outs["action"] = outs["action"].sample(seed=nj.rng())
        #     if getattr(self.config, "safe_eval", {}).get("enable", False):
        #         tau = float(self.config.safe_eval.get("tau", 0.20))
        #         if self.config.safe_eval.get("mode","one_step") == "long":
        #             outs["action"], outs["unsafe"], dbg = self.shield.apply_tau(latent, outs["action"], tau=tau)
        #         else:
        #             outs["action"], outs["unsafe"], dbg = self.shield.apply_tau(latent, outs["action"], tau=tau)
        #     outs["log_entropy"] = jnp.zeros(outs["action"].shape[:1])

        # inside Agent.policy(...):
        if mode == "eval":
            outs = task_outs
            outs["action"] = outs["action"].sample(seed=nj.rng())
            if getattr(self.config, "safe_eval", {}).get("enable", False):
                enforce = bool(self.config.safe_eval.get("enforce", False))
                tau  = float(self.config.safe_eval.get("tau", 0.30))

                # always compute risks for logging
                risks = self.eval_shield._risk_batch(latent)              # [B,N]
                orig_idx = jnp.argmax(outs["action"], -1)                 # [B]
                outs["log_orig_risk"] = risks[jnp.arange(risks.shape[0]), orig_idx]

                if enforce:
                    outs["action"], outs["unsafe"], dbg = self.eval_shield.apply_tau(tau=tau, latent=latent, action=outs["action"])
                    outs["log_shield_unsafe"]      = outs["unsafe"][:, None]
                    outs["log_shield_idx"]         = dbg["idx"]
                    outs["log_shield_orig_idx"]    = dbg["orig_idx"]
                    outs["log_shield_orig_risk"]   = dbg["orig_risk"]
                    outs["log_shield_chosen_risk"] = dbg["chosen_risk"]
                    outs["log_shield_lam"] = jnp.ones_like(outs["unsafe"]) * tau 
                else:
                    outs["unsafe"] = jnp.zeros_like(orig_idx, jnp.float32)

            outs["log_entropy"] = jnp.zeros(outs["action"].shape[:1])
            outs = _attach_risk_logs(outs, latent)

        elif mode == "explore":
            outs = expl_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=nj.rng())
            outs["action"], outs["unsafe"], dbg = self.shield.apply(latent, outs["action"], lam=lam, warmup=warm)

            # add non-action logs (driver filters them out before env.step)
            outs["log_shield_unsafe"]       = outs["unsafe"][:, None]
            outs["log_shield_idx"]          = dbg["idx"]
            outs["log_shield_orig_idx"]     = dbg["orig_idx"]
            outs["log_shield_orig_risk"]    = dbg["orig_risk"]
            outs["log_shield_chosen_risk"]  = dbg["chosen_risk"]
            # outs["log_shield_min_left"]     = dbg["min_left"]
            # outs["log_shield_min_right"]    = dbg["min_right"]
            outs["log_shield_lam"]         = jnp.ones_like(outs["unsafe"]) * lam
            outs = _attach_risk_logs(outs, latent)
        elif mode == "train":
            outs = task_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=nj.rng())
            outs["action"], outs["unsafe"], dbg = self.shield.apply(latent, outs["action"], lam=lam, warmup=warm)

            # add non-action logs (driver filters them out before env.step)
            outs["log_shield_unsafe"]       = outs["unsafe"][:, None]
            outs["log_shield_idx"]          = dbg["idx"]
            outs["log_shield_orig_idx"]     = dbg["orig_idx"]
            outs["log_shield_orig_risk"]    = dbg["orig_risk"]
            outs["log_shield_chosen_risk"]  = dbg["chosen_risk"]
            # outs["log_shield_min_left"]     = dbg["min_left"]
            # outs["log_shield_min_right"]    = dbg["min_right"]
            outs["log_shield_lam"]         = jnp.ones_like(outs["unsafe"]) * lam
            outs = _attach_risk_logs(outs, latent)
        # elif mode in ("explore", "train"):
        #     outs = expl_outs if mode == "explore" else task_outs
        #     outs["log_entropy"] = outs["action"].entropy()
        #     outs["action"] = outs["action"].sample(seed=nj.rng())
        #     outs["action"], outs["unsafe"], dbg = self.shield.apply(latent, outs["action"], lam=lam, warmup=warm)

        #     # add non-action logs (driver filters them out before env.step)
        #     outs["log_shield_unsafe"]       = outs["unsafe"][:, None]
        #     outs["log_shield_idx"]          = dbg["idx"]
        #     outs["log_shield_orig_idx"]     = dbg["orig_idx"]
        #     outs["log_shield_orig_risk"]    = dbg["orig_risk"]
        #     outs["log_shield_chosen_risk"]  = dbg["chosen_risk"]
        #     # outs["log_shield_min_left"]     = dbg["min_left"]
        #     # outs["log_shield_min_right"]    = dbg["min_right"]
        #     outs["log_shield_lam"]         = jnp.ones_like(outs["unsafe"]) * lam
          
        state = ((latent, outs["action"]), task_state, expl_state)
        return outs, state

    def train(self, data, state):
        self.config.jax.jit and print("Tracing train function.")
        metrics = {}
        data = self.preprocess(data)
        state, wm_outs, mets = self.wm.train(data, state)
        metrics.update(mets)
        context = {**data, **wm_outs["post"]}
        start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
        _, mets = self.task_behavior.train(self.wm.imagine, start, context)
        metrics.update(mets)
        if self.config.expl_behavior != "None":
            print(wm_outs.keys())
            if self.config.expl_rewards.rnd == 1.0:
                print("*******************************************************")
                print("using RND exploration reward")
                print("*******************************************************")
                data_for_expl = {**data, "embed": wm_outs["embed"]}
                _, mets = self.expl_behavior.train(self.wm.imagine, start, data_for_expl)
            else:
                _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
            metrics.update({"expl_" + key: value for key, value in mets.items()})

        if "keyA" in data.keys():
            outs = {
                "key": data["key"],
                "env_step": data["env_step"],
                "model_loss": metrics["model_loss_raw"].copy(),
                "td_error": metrics["td_error"].copy(),
            }

        else:
            outs = {}

        # Don't need the full model_loss_raw or td_error after the priority calculation, summarize it.
        metrics.update({"model_loss_raw": metrics["model_loss_raw"].mean()})
        metrics.update({"td_error": metrics["td_error"].mean()})

        return outs, state, metrics

    def report(self, data):
        self.config.jax.jit and print("Tracing report function.")
        data = self.preprocess(data)
        report = {}
        report.update(self.wm.report(data))
        mets = self.task_behavior.report(data)
        report.update({f"task_{k}": v for k, v in mets.items()})
        if self.expl_behavior is not self.task_behavior:
            mets = self.expl_behavior.report(data)
            report.update({f"expl_{k}": v for k, v in mets.items()})
        return report

    def preprocess(self, obs):
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_") or key in ("key", "env_step"):
                continue
            if len(value.shape) > 3 and value.dtype == jnp.uint8:
                value = jaxutils.cast_to_compute(value) / 255.0
            else:
                value = value.astype(jnp.float32)
            obs[key] = value

        if "lane_invasion" in obs:
            # >0 → off-lane at this step
            obs["offlane"] = (obs["lane_invasion"] > 0).astype(jnp.float32)

        if "collision" in obs:
            # >0 → off-lane at this step
            obs["collision"] = (obs["collision"] > 0).astype(jnp.float32)

        if "action" in obs and obs["action"].ndim == 4 and obs["action"].shape[2] == 1:
                obs["action"] = obs["action"].squeeze(2)
        obs["cont"] = 1.0 - obs["is_terminal"].astype(jnp.float32)
        return obs

class WorldModel(nj.Module):
    def __init__(self, obs_space, act_space, config):
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        shapes = {k: v for k, v in shapes.items() if not k.startswith("log_")}
        self.encoder = nets.MultiEncoder(shapes, **config.encoder, name="enc")
        self.rssm = nets.RSSM(**config.rssm, name="rssm")
        self.heads = {
            "decoder": nets.MultiDecoder(shapes, **config.decoder, name="dec"),
            "reward": nets.MLP((), **config.reward_head, name="rew"),
            "cont": nets.MLP((), **config.cont_head, name="cont"),
            "collision": nets.MLP((1,), **config.collision_head, name="collision"),
            "offlane": nets.MLP((1,), **config.offlane_head, name="offlane"),
        }
        self.opt = jaxutils.Optimizer(name="model_opt", **config.model_opt)
        scales = self.config.loss_scales.copy()
        image, vector = scales.pop("image"), scales.pop("vector")
        scales.update({k: image for k in self.heads["decoder"].cnn_shapes})
        scales.update({k: vector for k in self.heads["decoder"].mlp_shapes})
        self.scales = scales

    def initial(self, batch_size):
        prev_latent = self.rssm.initial(batch_size)
        prev_action = jnp.zeros((batch_size, *self.act_space.shape))
        return prev_latent, prev_action

    def train(self, data, state):
        modules = [self.encoder, self.rssm, *self.heads.values()]
        mets, (state, outs, metrics) = self.opt(modules, self.loss, data, state, has_aux=True)
        metrics.update(mets)
        return state, outs, metrics

    def loss(self, data, state):
        embed = self.encoder(data)
        prev_latent, prev_action = state
        prev_actions = jnp.concatenate([prev_action[:, None], data["action"][:, :-1]], 1)
        post, prior = self.rssm.observe(embed, prev_actions, data["is_first"], prev_latent)
        dists = {}
        feats = {**post, "embed": embed}
        for name, head in self.heads.items():
            out = head(feats if name in self.config.grad_heads else sg(feats))
            out = out if isinstance(out, dict) else {name: out}
            dists.update(out)
        losses = {}
        losses["dyn"] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
        losses["rep"] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
        for key, dist in dists.items():
            loss = -dist.log_prob(data[key].astype(jnp.float32))
            assert loss.shape == embed.shape[:2], (key, loss.shape)
            losses[key] = loss

        scaled = {k: v * self.scales[k] for k, v in losses.items()}
        model_loss = sum(scaled.values())
        out = {"embed": embed, "post": post, "prior": prior}
        out.update({f"{k}_loss": v for k, v in losses.items()})
        last_latent = {k: v[:, -1] for k, v in post.items()}
        last_action = data["action"][:, -1]
        state = last_latent, last_action
        metrics = self._metrics(data, dists, post, prior, losses, model_loss)
        metrics["model_loss_raw"] = model_loss  # Store model loss for Curious Replay prioritization
        return model_loss.mean(), (state, out, metrics)
    
    def _latent_to_embed(self, latent):
        """Predict pixels from a latent and re‑encode to 8 192‑D embed."""
        # latent → pixel prediction
        print("self.heads['decoder']:", self.heads["decoder"](latent))
        # obs_pred = self.heads["decoder"](latent)['birdeye_wpt'].mode()      # dict of tensors
        decoded = self.heads["decoder"](latent)
        obs_pred = {k: dist.mode() for k, dist in decoded.items()}

        # pixel tensors must carry a time axis for MultiEncoder -> add dummy T
        enc_in   = {k: v[:, None] for k, v in obs_pred.items()}   # [B, 1, ...]
        embed    = self.encoder(enc_in)[:, 0]                     # [B, 8192]

        return embed

    def imagine(self, policy, start, horizon):
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        start["action"] = policy(start)
        start["embed"]  = self._latent_to_embed(start)

        def step(prev, _):
            prev = prev.copy()
            state = self.rssm.img_step(prev, prev.pop("action"))
            return {**state, "action": policy(state), "embed": self._latent_to_embed(state)}

        traj = jaxutils.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
        cont = self.heads["cont"](traj).mode()
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def imagine_carry(self, policy, start, horizon, carry):
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        outs, carry = policy(start, carry)
        start["action"] = outs
        start["embed"]  = self._latent_to_embed(start)
        start["carry"] = carry

        def step(prev, _):
            prev = prev.copy()
            carry = prev.pop("carry")
            state = self.rssm.img_step(prev, prev.pop("action"))
            outs, carry = policy(state, carry)
            return {**state, "action": outs, "carry": carry, "embed": self._latent_to_embed(state)}

        traj = jaxutils.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items() if k != "carry"}
        cont = self.heads["cont"](traj).mode()
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def report(self, data):
        state = self.initial(len(data["is_first"]))
        # report = {}
        # report.update(self.loss(data, state)[-1][-1])
        # context, _ = self.rssm.observe(self.encoder(data)[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5])
        # start = {k: v[:, -1] for k, v in context.items()}
        # recon = self.heads["decoder"](context)
        # openl = self.heads["decoder"](self.rssm.imagine(data["action"][:6, 5:], start))

        report = {}
        B, T = data["is_first"].shape[:2]
        if T < 6:                                  # too short – skip visuals
            return report
        
        state  = self.initial(B)
        report.update(self.loss(data, state)[-1][-1])
        
        K = min(5, T - 1)                          # last recon frame index
        context, _ = self.rssm.observe(self.encoder(data)[:6, :K],data["action"][:6, :K],data["is_first"][:6, :K],)
        start  = {k: v[:, -1] for k, v in context.items()}
        recon  = self.heads["decoder"](context)
        openl  = self.heads["decoder"](self.rssm.imagine(data["action"][:6, K:], start))

        for key in self.heads["decoder"].cnn_shapes.keys():
            truth = data[key][:6].astype(jnp.float32)
            model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
            error = (model - truth + 1) / 2
            video = jnp.concatenate([truth, model, error], 2)
            report[f"openl_{key}"] = jaxutils.video_grid(video)
        return report

    def _metrics(self, data, dists, post, prior, losses, model_loss):
        entropy = lambda feat: self.rssm.get_dist(feat).entropy()
        metrics = {}
        metrics.update(jaxutils.tensorstats(entropy(prior), "prior_ent"))
        metrics.update(jaxutils.tensorstats(entropy(post), "post_ent"))
        metrics.update({f"{k}_loss_mean": v.mean() for k, v in losses.items()})
        metrics.update({f"{k}_loss_std": v.std() for k, v in losses.items()})
        metrics["model_loss_mean"] = model_loss.mean()
        metrics["model_loss_std"] = model_loss.std()
        metrics["reward_max_data"] = jnp.abs(data["reward"]).max()
        metrics["reward_max_pred"] = jnp.abs(dists["reward"].mean()).max()
        if "reward" in dists and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists["reward"], data["reward"], 0.1)
            metrics.update({f"reward_{k}": v for k, v in stats.items()})
        if "cont" in dists and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists["cont"], data["cont"], 0.5)
            metrics.update({f"cont_{k}": v for k, v in stats.items()})
        return metrics


class ImagActorCritic(nj.Module):
    def __init__(self, critics, scales, act_space, config):
        critics = {k: v for k, v in critics.items() if scales[k]}
        for key, scale in scales.items():
            assert not scale or key in critics, key
        self.critics = {k: v for k, v in critics.items() if scales[k]}
        self.scales = scales
        self.act_space = act_space
        self.config = config
        disc = act_space.discrete
        self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
        self.actor = nets.MLP(
            name="actor",
            dims="deter",
            shape=act_space.shape,
            **config.actor,
            dist=config.actor_dist_disc if disc else config.actor_dist_cont,
        )
        self.retnorms = {k: jaxutils.Moments(**config.retnorm, name=f"retnorm_{k}") for k in critics}
        self.opt = jaxutils.Optimizer(name="actor_opt", **config.actor_opt)

    def initial(self, batch_size):
        return {}

    def policy(self, state, carry):
        return {"action": self.actor(state)}, carry

    def train(self, imagine, start, context):
        def loss(start):
            policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
            traj = imagine(policy, start, self.config.imag_horizon)
            loss, metrics = self.loss(traj)
            return loss, (traj, metrics)

        mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
        metrics.update(mets)
        for key, critic in self.critics.items():
            mets = critic.train(traj, self.actor)
            metrics.update({f"{key}_critic_{k}": v for k, v in mets.items()})
        return traj, metrics

    def loss(self, traj):
        metrics = {}
        advs = []
        total = sum(self.scales[k] for k in self.critics)
        for key, critic in self.critics.items():
            rew, ret, base = critic.score(traj, self.actor)
            offset, invscale = self.retnorms[key](ret)
            normed_ret = (ret - offset) / invscale
            normed_base = (base - offset) / invscale
            advs.append((normed_ret - normed_base) * self.scales[key] / total)
            metrics.update(jaxutils.tensorstats(rew, f"{key}_reward"))
            metrics.update(jaxutils.tensorstats(ret, f"{key}_return_raw"))
            metrics.update(jaxutils.tensorstats(normed_ret, f"{key}_return_normed"))
            metrics[f"{key}_return_rate"] = (jnp.abs(ret) >= 0.5).mean()

        # if len(self.critics) != 1:
        #  raise NotImplementedError('Must have exactly one critic for TD error calculation.')

        r = jnp.reshape(rew[0], (self.config.batch_size, self.config.batch_length))
        v = jnp.reshape(base[0], (self.config.batch_size, self.config.batch_length))
        disc = jnp.reshape(traj["cont"][0], (self.config.batch_size, self.config.batch_length)) * (1 - 1 / self.config.horizon)
        td_error = r[:, :-1] + disc[:, 1:] * v[:, 1:] - v[:, :-1]
        metrics["td_error"] = td_error  # Store TD error for PER prioritization

        adv = jnp.stack(advs).sum(0)
        policy = self.actor(sg(traj))
        logpi = policy.log_prob(sg(traj["action"]))[:-1]
        loss = {"backprop": -adv, "reinforce": -logpi * sg(adv)}[self.grad]
        ent = policy.entropy()[:-1]
        loss -= self.config.actent * ent
        loss *= sg(traj["weight"])[:-1]
        loss *= self.config.loss_scales.actor
        metrics.update(self._metrics(traj, policy, logpi, ent, adv))
        return loss.mean(), metrics

    def _metrics(self, traj, policy, logpi, ent, adv):
        metrics = {}
        ent = policy.entropy()[:-1]
        rand = (ent - policy.minent) / (policy.maxent - policy.minent)
        rand = rand.mean(range(2, len(rand.shape)))
        act = traj["action"]
        act = jnp.argmax(act, -1) if self.act_space.discrete else act
        metrics.update(jaxutils.tensorstats(act, "action"))
        metrics.update(jaxutils.tensorstats(rand, "policy_randomness"))
        metrics.update(jaxutils.tensorstats(ent, "policy_entropy"))
        metrics.update(jaxutils.tensorstats(logpi, "policy_logprob"))
        metrics.update(jaxutils.tensorstats(adv, "adv"))
        metrics["imag_weight_dist"] = jaxutils.subsample(traj["weight"])
        return metrics


class VFunction(nj.Module):
    def __init__(self, rewfn, config):
        self.rewfn = rewfn
        self.config = config
        self.net = nets.MLP((), name="net", dims="deter", **self.config.critic)
        self.slow = nets.MLP((), name="slow", dims="deter", **self.config.critic)
        self.updater = jaxutils.SlowUpdater(
            self.net,
            self.slow,
            self.config.slow_critic_fraction,
            self.config.slow_critic_update,
        )
        self.opt = jaxutils.Optimizer(name="critic_opt", **self.config.critic_opt)

    def train(self, traj, actor):
        target = sg(self.score(traj)[1])
        mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
        metrics.update(mets)
        self.updater()
        return metrics

    def loss(self, traj, target):
        metrics = {}
        traj = {k: v[:-1] for k, v in traj.items()}
        dist = self.net(traj)
        loss = -dist.log_prob(sg(target))
        if self.config.critic_slowreg == "logprob":
            reg = -dist.log_prob(sg(self.slow(traj).mean()))
        elif self.config.critic_slowreg == "xent":
            reg = -jnp.einsum("...i,...i->...", sg(self.slow(traj).probs), jnp.log(dist.probs))
        else:
            raise NotImplementedError(self.config.critic_slowreg)
        loss += self.config.loss_scales.slowreg * reg
        loss = (loss * sg(traj["weight"])).mean()
        loss *= self.config.loss_scales.critic
        metrics = jaxutils.tensorstats(dist.mean())
        return loss, metrics

    def score(self, traj, actor=None):
        rew = self.rewfn(traj)
        assert len(rew) == len(traj["action"]) - 1, "should provide rewards for all but last action"
        discount = 1 - 1 / self.config.horizon
        disc = traj["cont"][1:] * discount
        value = self.net(traj).mean()
        vals = [value[-1]]
        interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
        for t in reversed(range(len(disc))):
            vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
        ret = jnp.stack(list(reversed(vals))[:-1])
        return rew, ret, value[:-1]
