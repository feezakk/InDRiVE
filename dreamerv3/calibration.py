# dreamerv3/calibration.py
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
import optax

EPS = 1e-6

def _logit(p):
  p = jnp.clip(p, EPS, 1 - EPS)
  return jnp.log(p) - jnp.log1p(-p)

def brier(probs, labels):
  p = np.asarray(probs, dtype=np.float32)
  y = np.asarray(labels, dtype=np.float32)
  return float(np.mean((p - y) ** 2))

def ece(probs, labels, n_bins=15):
  p = np.asarray(probs, dtype=np.float32)
  y = np.asarray(labels, dtype=np.float32)
  bins = np.linspace(0.0, 1.0, n_bins + 1)
  idx = np.digitize(p, bins) - 1
  e = 0.0
  N = len(p)
  for b in range(n_bins):
    m = idx == b
    if not np.any(m):
      continue
    conf = p[m].mean()
    acc = y[m].mean()
    e += (m.sum() / N) * abs(conf - acc)
  return float(e)

def roc_auc(probs, labels):
  p = np.asarray(probs, dtype=np.float64)
  y = np.asarray(labels, dtype=np.int32)
  n_pos = y.sum()
  n_neg = len(y) - n_pos
  if n_pos == 0 or n_neg == 0:
    return float("nan")
  order = np.argsort(p)
  ranks = np.empty_like(order)
  ranks[order] = np.arange(len(p)) + 1  # 1-based
  rank_sum_pos = ranks[y == 1].sum()
  auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
  return float(auc)

class TemperatureScaler:
  """p' = sigmoid((logit(p) - b) / T), learn T>0 and b."""
  def __init__(self):
    self.T = jnp.array(1.0, dtype=jnp.float32)
    self.b = jnp.array(0.0, dtype=jnp.float32)
    self.fitted = False

  def apply(self, probs):
    z = (_logit(probs) - self.b) / jnp.maximum(self.T, 1e-2)
    return jax.nn.sigmoid(z)

  def fit(self, probs, labels, steps=500, lr=0.05, seed=0):
    p = jnp.asarray(probs, dtype=jnp.float32)
    y = jnp.asarray(labels, dtype=jnp.float32)

    params = {"T": self.T, "b": self.b}
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    def loss_fn(params):
      z = (_logit(p) - params["b"]) / jnp.maximum(params["T"], 1e-2)
      q = jax.nn.sigmoid(z)
      # NLL (binary cross-entropy)
      nll = -jnp.mean(y * jnp.log(q + EPS) + (1 - y) * jnp.log(1 - q + EPS))
      return nll

    @jax.jit
    def step(params, opt_state):
      loss, grads = jax.value_and_grad(loss_fn)(params)
      updates, opt_state = opt.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      params = {"T": jnp.maximum(params["T"], 1e-2), "b": params["b"]}
      return params, opt_state, loss

    key = jax.random.PRNGKey(seed)
    for _ in range(steps):
      params, opt_state, _ = step(params, opt_state)

    self.T, self.b = params["T"], params["b"]
    self.fitted = True

class CalibManager:
  def __init__(self, hazards=("collision","off_road","wrong_direction"),
               n_bins=15, min_total=5000, min_pos=200, update_every=5000):
    self.hazards = list(hazards)
    self.n_bins = n_bins
    self.min_total = int(min_total)
    self.min_pos = int(min_pos)
    self.update_every = int(update_every)
    self.buf = {h: {"p": [], "y": []} for h in self.hazards}
    self.scaler = {h: TemperatureScaler() for h in self.hazards}
    self._last_update = -1

  def collect(self, hazard, probs, labels):
    if hazard not in self.buf:  # ignore unknown
      return
    self.buf[hazard]["p"].append(np.asarray(probs).reshape(-1))
    self.buf[hazard]["y"].append(np.asarray(labels).reshape(-1))

  def _ready(self, hazard):
    p = np.concatenate(self.buf[hazard]["p"], axis=0) if self.buf[hazard]["p"] else np.array([])
    y = np.concatenate(self.buf[hazard]["y"], axis=0) if self.buf[hazard]["y"] else np.array([])
    return (len(p) >= self.min_total) and (y.sum() >= self.min_pos)

  def fit_and_log_if_ready(self, step, logger_print=print):
    if self._last_update >= 0 and (step - self._last_update) < self.update_every:
      return
    did_any = False
    for h in self.hazards:
      if not self._ready(h):
        continue
      p = np.concatenate(self.buf[h]["p"], axis=0)
      y = np.concatenate(self.buf[h]["y"], axis=0).astype(np.int32)

      # holdout split
      n = len(p)
      idx = np.random.RandomState(0).permutation(n)
      tr = idx[: int(0.7 * n)]
      va = idx[int(0.7 * n):]

      self.scaler[h].fit(p[tr], y[tr], steps=500, lr=0.05)

      p_raw = p[va]
      p_cal = np.array(self.scaler[h].apply(jnp.asarray(p_raw)))
      y_va = y[va]

      metrics = dict(
        count=len(p_raw),
        pos=int(y_va.sum()),
        ECE_raw=ece(p_raw, y_va, self.n_bins),
        Brier_raw=brier(p_raw, y_va),
        AUC_raw=roc_auc(p_raw, y_va),
        ECE_cal=ece(p_cal, y_va, self.n_bins),
        Brier_cal=brier(p_cal, y_va),
        AUC_cal=roc_auc(p_cal, y_va),
      )
      logger_print(
        f"[Calib/{h}] n={metrics['count']} pos={metrics['pos']}  "
        f"ECE raw={metrics['ECE_raw']:.3f} → cal={metrics['ECE_cal']:.3f}  "
        f"Brier raw={metrics['Brier_raw']:.3f} → cal={metrics['Brier_cal']:.3f}  "
        f"ROC-AUC raw={metrics['AUC_raw']:.3f} → cal={metrics['AUC_cal']:.3f}"
      )
      did_any = True
    if did_any:
      self._last_update = step

  def apply(self, hazard, probs):
    if hazard in self.scaler and self.scaler[hazard].fitted:
      return self.scaler[hazard].apply(probs)
    return probs
