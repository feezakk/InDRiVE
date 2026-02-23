# tools/eval_metrics.py
import json, math
from pathlib import Path
import numpy as np

# ---------- utilities ----------
def _to1d(x): return np.asarray(x).reshape(-1)
def _align_len(x, n):
    x = _to1d(x)
    if x.size >= n: return x[:n]
    if x.size == 0: return np.zeros(n, dtype=float)
    return np.concatenate([x, np.repeat(x[-1], n - x.size)])

def _sigmoid(z):
    z = np.asarray(z, np.float32)
    z = np.clip(z, -20.0, 20.0)          # prevents overflow in exp
    return 1.0 / (1.0 + np.exp(-z))

def _event_on_step(x):
    x = np.asarray(x)
    if x.size == 0: return x.astype(np.int32)
    x = (x > 0).astype(np.int32)
    d = np.diff(np.concatenate([[0], x]))
    return (d > 0).astype(np.int32)

def _ece_brier(p, y, n_bins=15):
    p = np.clip(np.asarray(p).ravel(), 1e-6, 1-1e-6)
    y = np.asarray(y).astype(np.float64).ravel()
    bins = np.linspace(0., 1., n_bins + 1)
    idx  = np.clip(np.digitize(p, bins) - 1, 0, n_bins-1)
    cnt  = np.bincount(idx, minlength=n_bins).astype(np.float64)
    acc  = np.zeros(n_bins); conf = np.zeros(n_bins)
    for b in range(n_bins):
        if cnt[b] > 0:
            m = (idx == b)
            acc[b]  = y[m].mean()
            conf[b] = p[m].mean()
        else:
            acc[b] = np.nan; conf[b] = np.nan
    w   = cnt / max(1.0, cnt.sum())
    ece = float(np.nansum(np.abs(acc - conf) * w))
    brier = float(np.mean((p - y) ** 2))
    # also return points for reliability curve
    mids = 0.5 * (bins[:-1] + bins[1:])
    return ece, brier, mids, acc, conf, cnt

def _risk_coverage(p, y):
    """Coverage vs risk (hazard rate) when including low-risk steps first."""
    p = np.asarray(p).ravel(); y = np.asarray(y).astype(np.int32).ravel()
    order = np.argsort(p)                  # low risk -> included first
    y = y[order]
    n = len(y); k = np.arange(1, n+1)
    cov = k / n
    risk = np.cumsum(y) / k
    aurc = float(np.trapz(risk, cov))
    return cov, risk, aurc

def _load_calibrator(path):
    """Expect keys: collision_T, collision_b, off_road_T, off_road_b."""
    if not path or not Path(path).exists(): return {}
    arr = np.load(path)
    def _mk(T, b):
        T = float(T); b = float(b)
        def apply(p):
            p = np.clip(np.asarray(p, np.float64), 1e-6, 1-1e-6)
            logit = np.log(p) - np.log1p(-p)
            z = (logit - b) / max(T, 1e-6)
            return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
        return apply
    cal = {}
    if "collision_T" in arr and "collision_b" in arr:
        cal["collision"] = _mk(arr["collision_T"], arr["collision_b"])
    if "off_road_T" in arr and "off_road_b" in arr:
        cal["off_road"] = _mk(arr["off_road_T"], arr["off_road_b"])
    return cal

# ---------- recorder ----------
class EvalMetrics:
    """
    Drop-in for eval: call  metrics = EvalMetrics(outdir, fps, agg, tau, calibrator_path)
    and register: driver.on_episode(metrics.on_episode)
    """
    def __init__(self, outdir, fps=10.0, agg="max", tau=0.30, calibrator_path=None):
        self.outdir = Path(outdir); (self.outdir / "eval_dump").mkdir(parents=True, exist_ok=True)
        self.jsonl  = open(self.outdir / "eval_dump" / "episodes.jsonl", "a", buffering=1)
        self.fps = float(fps); self.dt = 1.0 / self.fps
        self.agg = str(agg).lower()
        self.tau = float(tau)
        self.cal = _load_calibrator(calibrator_path)
        self.idx = 0

    def _combine_any(self, a, b):
        if self.agg == "product": return 1.0 - (1.0 - a) * (1.0 - b)
        if self.agg == "mean":    return 0.5 * (a + b)
        return np.maximum(a, b)   # "max"

    def on_episode(self, ep, ep_info, worker=None):
        # --- gather per-step ---
        p_col_raw = _to1d(ep.get("log_p_collision", []))
        p_off_raw = _align_len(ep.get("log_p_offlane",   []), len(p_col_raw))
        n = len(p_col_raw)

        y_col = _event_on_step(_align_len(ep.get("collision", []), n))
        y_off = _event_on_step(_align_len(ep.get("lane_invasion", ep.get("offlane", [])), n))
        y_any = np.clip(y_col + y_off, 0, 1)

        unsafe = _align_len(ep.get("unsafe", np.zeros(n)), n) > 0.5
        speed  = _align_len(ep.get("speed_mps", np.zeros(n)), n)

        # calibrated
        q_col = self.cal.get("collision", lambda x: x)(p_col_raw)
        q_off = self.cal.get("off_road",  lambda x: x)(p_off_raw)
        p_any = self._combine_any(p_col_raw, p_off_raw)
        q_any = self._combine_any(q_col,     q_off)

        # ---- episode‑level stats ----
        dist_m = float(ep_info.get("meters", speed.sum() * self.dt))
        hazards = int(y_col.sum() + y_off.sum())
        haz_per_km = hazards / max(dist_m / 1000.0, 1e-6)
        overrides_pct = float(unsafe.mean() * 100.0)

        # success flag if available
        term = str(ep_info.get("terminal_reason", "")).lower()
        success = bool(ep_info.get("success", ("success" in term)))
        ep_return = float(np.asarray(ep.get("reward", []), np.float64).sum())

        # lead-time: first time q_any >= tau before each event start
        leads = []
        idxs = np.where(y_any > 0)[0]
        for t in idxs:
            prev = np.where(q_any[:t+1] >= self.tau)[0]
            if prev.size:
                steps = int(t - prev[-1])
                leads.append(steps * self.dt)
        lead_time_mean = float(np.mean(leads)) if leads else 0.0
        lead_time_p90  = float(np.quantile(leads, 0.90)) if leads else 0.0

        # calibration metrics (dataset-level to be aggregated later; keep per-ep too)
        def cal_block(p, y, tag):
            e, b, mids, acc, conf, cnt = _ece_brier(p, y)
            return {f"{tag}_ece": e, f"{tag}_brier": b, f"{tag}_pos": int(y.sum()),
                    f"{tag}_bins": mids.tolist(), f"{tag}_acc": np.nan_to_num(acc).tolist(),
                    f"{tag}_conf": np.nan_to_num(conf).tolist(), f"{tag}_cnt": cnt.astype(int).tolist()}

        cal_raw = {}
        cal_raw.update(cal_block(p_col_raw, y_col, "raw_col"))
        cal_raw.update(cal_block(p_off_raw, y_off, "raw_off"))
        cal_raw.update(cal_block(p_any,     y_any, "raw_any"))

        cal_cal = {}
        cal_cal.update(cal_block(q_col, y_col, "cal_col"))
        cal_cal.update(cal_block(q_off, y_off, "cal_off"))
        cal_cal.update(cal_block(q_any, y_any, "cal_any"))

        # risk–coverage (on ANY, raw & cal)
        cov_r, risk_r, aurc_r = _risk_coverage(p_any, y_any)
        cov_c, risk_c, aurc_c = _risk_coverage(q_any, y_any)

        # save per‑episode arrays for later plotting if desired
        npz_path = self.outdir / "eval_dump" / f"ep_{self.idx:05d}.npz"
        np.savez(npz_path,
                 p_col=p_col_raw, p_off=p_off_raw, p_any=p_any,
                 q_col=q_col,     q_off=q_off,     q_any=q_any,
                 y_col=y_col,     y_off=y_off,     y_any=y_any,
                 unsafe=unsafe.astype(np.int8), speed=speed.astype(np.float32),
                 dt=np.array(self.dt, np.float32), tau=np.array(self.tau, np.float32),
                 cov_raw=cov_r, risk_raw=risk_r, cov_cal=cov_c, risk_cal=risk_c)

        row = {
            "ep_idx": self.idx,
            "npz": npz_path.name,
            "success": success,
            "hazards": hazards,
            "distance_m": dist_m,
            "hazards_per_km": haz_per_km,
            "overrides_pct": overrides_pct,
            "lead_mean_s": lead_time_mean,
            "lead_p90_s":  lead_time_p90,
            "return": ep_return,
            "aurc_raw": aurc_r,
            "aurc_cal": aurc_c,
            **cal_raw, **cal_cal
        }
        self.jsonl.write(json.dumps(row) + "\n")
        self.idx += 1

    def close(self):
        try: self.jsonl.close()
        except Exception: pass
