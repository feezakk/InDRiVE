import re

import embodied
import jax
import numpy as np
import jax, jax.numpy as jnp
import csv
from dreamerv3 import ninjax as nj
from dreamerv3 import shield_bus as sb
from collections import deque
import numpy as np
from sklearn.metrics import roc_auc_score  # if not already imported
# from calibration import CalibManager

from dreamerv3.calibration import CalibManager
import csv, atexit

def _to1d(x):
    return np.asarray(x).reshape(-1)

def _event_on_step(x):
    x = np.asarray(x)
    if x.size == 0:
        return x.astype(np.int32)
    x = (x > 0).astype(np.int32)                # cumulative -> {0,1}
    d = np.diff(np.concatenate([[0], x]))       # 1 only when it turns on
    return (d > 0).astype(np.int32)

def _align_len(x, n):
    x = _to1d(x)
    if x.size >= n:
        return x[:n]
    if x.size == 0:
        return np.zeros(n, dtype=float)
    return np.concatenate([x, np.repeat(x[-1], n - x.size)])

def _get_stream(ep, ep_info, keys, n):
    for k in keys:
        if k in ep:
            return (_align_len(ep[k], n) > 0).astype(np.float32)
        if k in ep_info:
            return (_align_len(ep_info[k], n) > 0).astype(np.float32)
    return np.zeros(n, np.float32)


def _ece_brier(probs, labels, n_bins=15):
    probs = np.asarray(probs).astype(np.float64).ravel()
    labels = np.asarray(labels).astype(np.float64).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins).astype(np.float64)
    acc = np.zeros(n_bins); conf = np.zeros(n_bins)
    for b in range(n_bins):
        if cnt[b] > 0:
            m = (idx == b)
            acc[b] = labels[m].mean()
            conf[b] = probs[m].mean()
        else:
            acc[b] = np.nan; conf[b] = np.nan
    w = cnt / max(1.0, cnt.sum())
    ece = np.nansum(np.abs(acc - conf) * w)
    brier = float(np.mean((probs - labels) ** 2))
    return float(ece), brier

def _roc_auc(probs, labels):
    p = np.asarray(probs).ravel()
    y = np.asarray(labels).astype(np.int32).ravel()
    if y.sum() == 0 or y.sum() == len(y): return np.nan
    o = np.argsort(-p); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = tp / (tp[-1] if tp[-1] > 0 else 1); fpr = fp / (fp[-1] if fp[-1] > 0 else 1)
    tpr = np.concatenate([[0.0], tpr, [1.0]]); fpr = np.concatenate([[0.0], fpr, [1.0]])
    return float(np.trapz(tpr, fpr))

def _pr_auc(probs, labels):
    p = np.asarray(probs).ravel()
    y = np.asarray(labels).astype(np.int32).ravel()
    if y.sum() == 0: return np.nan
    o = np.argsort(-p); y = y[o]
    tp = np.cumsum(y).astype(np.float64); fp = np.cumsum(1 - y).astype(np.float64)
    prec = tp / np.maximum(tp + fp, 1.0); rec = tp / (tp[-1] if tp[-1] > 0 else 1.0)
    prec = np.concatenate([[1.0], prec]); rec = np.concatenate([[0.0], rec])
    return float(np.trapz(prec, rec))



def train(agent, env, replay, shield, logger, args):
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    # Optional CSV logging for per-step training traces
    train_csv_file = None
    train_csv_writer = None
    if getattr(args, "log_per_step", False):
        try:
            train_csv_path = str(logdir / "train_steps.csv")
            train_csv_file = open(train_csv_path, "w", newline="")
            # fieldnames = ["step", "env_step", "reward", "unsafe", "action"]
            fieldnames = [
                "step", "env_step", "reward", "unsafe",
                "action",
                "speed_ms",
                "comfort_acc_ms2", "comfort_jerk_ms3",
                "comfort_dsteer_abs", "comfort_dthrottle_abs",
                "comfort_lat_acc_ms2",
                "traffic_density",
            ]
            train_csv_writer = csv.DictWriter(train_csv_file, fieldnames=fieldnames)
            train_csv_writer.writeheader()
        except Exception:
            train_csv_file = None
            train_csv_writer = None
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Clock(args.save_every)
    should_sync = embodied.when.Every(args.sync_every)
    step = logger.step
    updates = embodied.Counter()
    metrics = embodied.Metrics()
    print("Observation space:", embodied.format(env.obs_space), sep="\n")
    print("Action space:", embodied.format(env.act_space), sep="\n")

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("env", env, ["step"])
    timer.wrap("replay", replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    best_ckpt = embodied.Checkpoint(logdir / "checkpoint_best.ckpt")
    timer.wrap("checkpoint_best", best_ckpt, ["save", "load"])
    best_ckpt.step = step
    best_ckpt.agent = agent
    # (omit replay to keep file small)
    best_ext_return = -np.inf
    
    calib = CalibManager(
        hazards=("collision","off_road"),
        n_bins=15, min_total=800, min_pos=20, update_every=500
        )

    # --- Calibration buffers & helpers (scoped to one training run) ---
    CALIB_BUF_OFF = deque(maxlen=5000)      # (p, y) per step for off-road
    CALIB_BUF_COL = deque(maxlen=5000)      # (p, y) per step for collision

    def pick_tau(p, y, target_fpr=0.05):
        yb = y.astype(bool)
        neg = p[~yb]
        if len(neg) == 0:
            return 1.0
        return float(np.quantile(neg, 1.0 - target_fpr))

    def recompute_threshold(buf, target_fpr=0.05):
        if len(buf) < 500:
            return None
        arr = np.array(buf, dtype=np.float32)
        pp, yy = arr[:, 0], arr[:, 1].astype(bool)
        # auto-flip if ROC < 0.5 and both classes exist
        roc = roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan
        if not np.isnan(roc) and roc < 0.5:
            pp = 1.0 - pp
        return pick_tau(pp, yy, target_fpr)
    

    def _make_writer(path, fieldnames):
        path = embodied.Path(path)
        exists = path.exists()
        f = open(str(path), "a", newline="")
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
            f.flush()
        return f, w

    EP_FIELDS = [
        "episode_index", "env_step", "length", "return",

        # outcome flags (failure breakdown)
        "success",
        "collision", "off_road", "out_of_lane", "wrong_direction", "too_slow", "time_exceeded", "destination_reached",

        # speed tracking (reviewer request)
        "mean_speed_ms", "std_speed_ms",

        # comfort/smoothness (reviewer request)
        "mean_abs_acc_ms2", "mean_abs_jerk_ms3",
        "mean_abs_dsteer", "mean_abs_dthrottle",
        "mean_abs_lat_acc_ms2",

        # optional context (only if present in info; will be NA otherwise)
        "lane_pair_index", "traffic_density",
    ]

    train_csv_f, train_csv_w = _make_writer(logdir / "train_episode_metrics.csv", EP_FIELDS)
    atexit.register(lambda: train_csv_f.close())
    train_ep_idx = {"v": 0}

    train_csv_f, train_csv_w = _make_writer(logdir / "train_success.csv")
    eval_csv_f,  eval_csv_w  = _make_writer(logdir / "eval_success.csv")
    atexit.register(lambda: (train_csv_f.close(), eval_csv_f.close()))
    train_ep_idx = {"v": 0}
    eval_ep_idx  = {"v": 0}

    def _any_key(ep, ep_info, keys):
        for k in keys:
            if k in ep_info:
                v = np.asarray(ep_info[k])
                if v.size and np.any(v):
                    return True
            if k in ep:
                v = np.asarray(ep[k])
                if v.size and np.any(v):
                    return True
        return False

    def _series(ep, ep_info, keys, n, default=0.0):
        for k in keys:
            if k in ep:
                return _align_len(ep[k], n).astype(np.float32)
            if k in ep_info:
                return _align_len(ep_info[k], n).astype(np.float32)
        return np.full((n,), float(default), np.float32)

    def _safe_mean_abs(x):
        x = np.asarray(x, np.float64).reshape(-1)
        return float(np.mean(np.abs(x))) if x.size else 0.0

    def _safe_mean(x):
        x = np.asarray(x, np.float64).reshape(-1)
        return float(np.mean(x)) if x.size else 0.0

    def _safe_std(x):
        x = np.asarray(x, np.float64).reshape(-1)
        return float(np.std(x)) if x.size else 0.0

    def _csv_log(ep, ep_info):
        length = int(len(ep["reward"]) - 1)
        ret = float(ep["reward"].astype(np.float64).sum())
        n = max(1, length)  # avoid zero-length

        collision = int(_any_key(ep, ep_info, ["is_collision", "collision"]))
        off_road = int(_any_key(ep, ep_info, ["is_off_road", "off_road"]))
        out_of_lane = int(_any_key(ep, ep_info, ["out_of_lane", "lane_invasion", "offlane"]))
        wrong_direction = int(_any_key(ep, ep_info, ["is_wrong_direction", "wrong_direction"]))
        too_slow = int(_any_key(ep, ep_info, ["too_slow", "not_moving"]))
        time_exceeded = int(_any_key(ep, ep_info, ["time_exceeded"]))
        destination_reached = int(_any_key(ep, ep_info, ["is_destination_reached", "destination_reached", "goal_reached"]))

        success = int(
            (destination_reached == 1)
            and (collision == 0)
            and (off_road == 0)
            and (out_of_lane == 0)
            and (wrong_direction == 0)
            and (too_slow == 0)
            and (time_exceeded == 0)
        )

        # Per-step series for comfort/speed (these are added in env.step in §1/§2)
        speed_ms = _series(ep, ep_info, ["speed_ms", "speed_norm"], n, default=0.0)
        acc_ms2 = _series(ep, ep_info, ["comfort_acc_ms2"], n, default=0.0)
        jerk_ms3 = _series(ep, ep_info, ["comfort_jerk_ms3"], n, default=0.0)
        dsteer = _series(ep, ep_info, ["comfort_dsteer_abs"], n, default=0.0)
        dthr = _series(ep, ep_info, ["comfort_dthrottle_abs"], n, default=0.0)
        latacc = _series(ep, ep_info, ["comfort_lat_acc_ms2"], n, default=0.0)

        # Context (optional)
        lane_pair = ep_info.get("lane_pair_index", ep.get("lane_pair_index", "NA"))
        dens = ep_info.get("traffic_density", ep.get("traffic_density", "NA"))

        row = {
            "episode_index": int(train_ep_idx["v"]),
            "env_step": int(logger.step),
            "length": int(length),
            "return": float(ret),

            "success": int(success),
            "collision": int(collision),
            "off_road": int(off_road),
            "out_of_lane": int(out_of_lane),
            "wrong_direction": int(wrong_direction),
            "too_slow": int(too_slow),
            "time_exceeded": int(time_exceeded),
            "destination_reached": int(destination_reached),

            "mean_speed_ms": _safe_mean(speed_ms),
            "std_speed_ms": _safe_std(speed_ms),

            "mean_abs_acc_ms2": _safe_mean_abs(acc_ms2),
            "mean_abs_jerk_ms3": _safe_mean_abs(jerk_ms3),
            "mean_abs_dsteer": _safe_mean_abs(dsteer),
            "mean_abs_dthrottle": _safe_mean_abs(dthr),
            "mean_abs_lat_acc_ms2": _safe_mean_abs(latacc),

            "lane_pair_index": lane_pair if isinstance(lane_pair, (int, float, str)) else "NA",
            "traffic_density": dens if isinstance(dens, (int, float, str)) else "NA",
        }

        train_csv_w.writerow(row)
        train_csv_f.flush()
        train_ep_idx["v"] += 1

        

    def per_episode(ep,ep_info):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        success = float(np.any(np.array(ep_info.get("goal_reached", [0]))))
        collision = float(np.any(np.array(ep_info.get("collision", [0]))))
        past_goal = float(np.any(np.array(ep_info.get("past_goal", [0]))))
        not_moving = float(np.any(np.array(ep_info.get("not_moving", [0]))))
        time_exceeded = float(np.any(np.array(ep_info.get("time_exceeded", [0]))))

        sum_abs_reward = float(np.abs(ep["reward"]).astype(np.float64).sum())
        logger.add(
            {
                "length": length,
                "score": score,
                "sum_abs_reward": sum_abs_reward,
                "reward_rate": (np.abs(ep["reward"]) >= 0.5).mean(),
                "success": success,
                "collision": collision,
                "past_goal": past_goal,
                "not_moving": not_moving,
                "time_exceeded": time_exceeded

            },
            prefix="episode",
        )

        if "log_p_collision" in ep or "v" in ep:
            # Align all arrays to the same length n.
            n = len(ep.get("log_p_collision", ep.get("log_p_offlane")))
            def _al(a): return _align_len(a, n)

            # Labels: 1 only on the step the event *starts*.
            y_col = _event_on_step(_al(ep.get("collision", [])))           # [n]
            # off-road label can come from lane_invasion or your offlane flag
            y_off = _event_on_step(_al(ep.get("lane_invasion", ep.get("offlane", []))))

            # Model probabilities written by Agent.policy()
            if "log_p_collision" in ep:
                p_col = _al(ep["log_p_collision"]).astype(np.float32).reshape(-1)
                calib.collect("collision", p_col, y_col)

                 # ---- quick collision diagnostics (calibrated) ----
                q_col = np.asarray(calib.apply("collision", p_col), np.float32).reshape(-1)

                pos = int(y_col.sum()); tot = int(len(y_col))
                ece_c, brier_c = _ece_brier(q_col, y_col, n_bins=15)
                roc_c = _roc_auc(q_col, y_col) if (0 < pos < tot) else np.nan  # guard
                pr_c  = _pr_auc(q_col, y_col)  if (pos > 0) else np.nan        # guard

                logger.add({"col_ece_cal": float(ece_c),
                            "col_brier_cal": float(brier_c)}, prefix="calib")
                if not np.isnan(roc_c):
                    logger.add({"col_roc_auc_cal": float(roc_c)}, prefix="calib")
                if not np.isnan(pr_c):
                    logger.add({"col_pr_auc_cal": float(pr_c)}, prefix="calib")

                print(f"[col] pos={pos} / total={tot}  "
                    f"ECE={ece_c:.3f}  Brier={brier_c:.3f}  "
                    f"ROC={'nan' if np.isnan(roc_c) else f'{roc_c:.3f}'}  "
                    f"PR={'nan' if np.isnan(pr_c) else f'{pr_c:.3f}'}")

            if "log_p_offlane" in ep:
                p_off = _al(ep["log_p_offlane"]).astype(np.float32).reshape(-1)
                calib.collect("off_road",  p_off, y_off)

            # Periodically fit + print metrics (uses internal holdout)
            calib.fit_and_log_if_ready(step=int(step), logger_print=print)

        try:
            cal_path = str(logdir / "calibrator.npz")
            payload = {}
            for h in ("collision", "off_road"):
                sc = calib.scaler.get(h, None)
                if sc and getattr(sc, "fitted", False):
                    payload[f"{h}_T"] = np.array(sc.T)
                    payload[f"{h}_b"] = np.array(sc.b)
            if payload:  # save whatever is fitted so far
                np.savez(cal_path, **payload)
                print(f"[Calib] Saved calibrator to {cal_path}")
        except Exception as e:
            print(f"[Calib] Save skipped: {e}")

        # # ---- Calibration over this episode (one-step risk) ----
        # if "log_p_risk" in ep:
        #     p = _to1d(ep["log_p_risk"])
        #     n = p.size

        #     # Try multiple possible keys; use what exists and align lengths.
        #     col = _get_stream(ep, ep_info, ["collision", "is_collision"], n)
        #     off = _get_stream(ep, ep_info,
        #                     ["offlane", "lane_invasion", "out_of_lane", "is_off_road", "is_wrong_direction"],
        #                     n)
        #     y = np.clip(col + off, 0, 1)  # union

        #     ece, brier = _ece_brier(p, y, n_bins=15)
        #     roc = _roc_auc(p, y)
        #     pr  = _pr_auc(p, y)

        #     logger.add({"ece": ece, "brier": brier, "roc_auc": roc, "pr_auc": pr}, prefix="calib")
        #     print(f"[Calib] len(p)={n} len(col)={len(col)} len(off)={len(off)} "
        #         f"ECE={ece:.3f} Brier={brier:.3f} ROC-AUC={roc:.3f} PR-AUC={pr:.3f}")
            
        #     # if (y.sum() > 0) and (y.sum() < len(y)):
        #     #     pos = float(p[y].mean()); neg = float(p[~y].mean())
        #     #     logger.add({"pos_mean": pos, "neg_mean": neg}, prefix="calib")
        #     #     print(f"[Calib] pos_mean={pos:.3f}  neg_mean={neg:.3f}")
                
        #     p = np.asarray(ep["log_p_risk"]).reshape(-1).astype(np.float32)
        #     col = np.asarray(ep.get("collision", np.zeros(len(p)))).reshape(-1)[:len(p)] > 0
        #     off = np.asarray(ep.get("offlane",   np.zeros(len(p)))).reshape(-1)[:len(p)] > 0
        #     y = (col | off)                      # boolean mask

        #     # Avoid previous crash:
        #     if y.any() and (~y).any():
        #         pos_mean = float(p[y].mean()); neg_mean = float(p[~y].mean())
        #         logger.add({"pos_mean": pos_mean, "neg_mean": neg_mean}, prefix="calib")
        #         print(f"[Calib] pos_mean={pos_mean:.3f}  neg_mean={neg_mean:.3f}")

        #     p = np.asarray(p, dtype=np.float32).reshape(-1)
        #     y = np.asarray(y).astype(bool).reshape(-1)

        #     # Drop non-finite preds
        #     mask = np.isfinite(p)
        #     p, y = p[mask], y[mask]

        #     roc = np.nan
        #     try:
        #         if 0 < y.sum() < len(y):
        #             roc = roc_auc_score(y, p)
        #             # Auto-flip if ranking is inverted
        #             flipped = (not np.isnan(roc)) and (roc < 0.5)
        #             if flipped:
        #                 p = 1.0 - p
        #                 roc = 1.0 - roc
        #             pos = float(p[y].mean())
        #             neg = float(p[~y].mean())
        #             logger.add({"pos_mean": pos, "neg_mean": neg, "calib_flipped": float(flipped)}, prefix="calib")
        #     except Exception as e:
        #         print(f"[Calib] Safe metrics failed: {e}")

        #     # Ensure boolean indexing (fixes the IndexError)
        #     yb = y.astype(bool)

        #     # (Optional) keep your pos/neg means safely
        #     if 0 < yb.sum() < len(yb):
        #         pos = float(p[yb].mean())
        #         neg = float(p[~yb].mean())
        #         logger.add({"pos_mean": pos, "neg_mean": neg}, prefix="calib")
        #         print(f"[Calib] pos_mean={pos:.3f}  neg_mean={neg:.3f}")

        #     # Determine hazard type for this episode
        #     term = str(ep_info.get("terminal_reason", "")).lower()
        #     hazard_type = "offroad" if "off" in term else ("collision" if "coll" in term else None)

        #     # Append (p, y) to the appropriate calibration buffer
        #     # if hazard_type == "offroad":
        #     #     CALIB_BUF_OFF.extend(zip(p.tolist(), yb.astype(np.uint8).tolist()))
        #     # elif hazard_type == "collision":
        #     #     CALIB_BUF_COL.extend(zip(p.tolist(), yb.astype(np.uint8).tolist()))

        #     CALIB_BUF_OFF.extend(zip(p.tolist(), off.astype(np.uint8).tolist()))
        #     CALIB_BUF_COL.extend(zip(p.tolist(), col.astype(np.uint8).tolist()))

        #     # Recompute thresholds (guarded by len>=500) and log them
        #     tau_off = recompute_threshold(CALIB_BUF_OFF, target_fpr=0.05)
        #     tau_col = recompute_threshold(CALIB_BUF_COL, target_fpr=0.05)
        #     if tau_off is not None:
        #         logger.add({"tau_off": float(tau_off)}, prefix="calib")
        #     if tau_col is not None:
        #         logger.add({"tau_col": float(tau_col)}, prefix="calib")

        #     # (Optional) if you have a shield/barrier object, push the new τ values into it
        #     # if shield is not None and hasattr(shield, "set_threshold"):
        #     #     if tau_off is not None: shield.set_threshold("offroad", tau_off)
        #     #     if tau_col is not None: shield.set_threshold("collision", tau_col)




        # # print("[reward]", ep_info["ext_reward_step"])

        # ---- Calibrated metrics (hazard-wise + combined) ----
        n = len(ep.get("log_p_collision", ep.get("log_p_offlane", [])))


        if n == 0:
            return  # nothing to calibrate/log this episode

        # labels: event turns on at that step
        y_col = _event_on_step(_align_len(ep.get("collision", []), n))
        y_off = _event_on_step(_align_len(ep.get("lane_invasion", ep.get("offlane", [])), n))
        y_any = np.clip(y_col + y_off, 0, 1)

        # raw model probs from policy() logs
        p_col_raw = _align_len(ep.get("log_p_collision", np.zeros(n)), n).astype(np.float32)
        p_off_raw = _align_len(ep.get("log_p_offlane",   np.zeros(n)), n).astype(np.float32)

        # apply saved/online calibrator
        p_col_cal = np.asarray(calib.apply("collision", p_col_raw))
        p_off_cal = np.asarray(calib.apply("off_road",  p_off_raw))

        # combine calibrated risks to match your eval agg (you used --safe_eval.agg max)
        agg = str(getattr(getattr(args, "safe_eval", {}), "agg", "max")).lower()
        if agg == "product":
            p_any_cal = 1.0 - (1.0 - p_col_cal) * (1.0 - p_off_cal)
        elif agg == "mean":
            p_any_cal = 0.5 * (p_col_cal + p_off_cal)
        else:  # "max"
            p_any_cal = np.maximum(p_col_cal, p_off_cal)

        # metrics (calibrated)
        def _log_cal(name, p, y):
            e, b = _ece_brier(p, y, n_bins=15); r = _roc_auc(p, y); pr = _pr_auc(p, y)
            logger.add({f"{name}_ece2_cal": e, f"{name}_brier2_cal": b,
                        f"{name}_roc_auc2_cal": r, f"{name}_pr_auc2_cal": pr}, prefix="calib")

        _log_cal("col", p_col_cal, y_col)
        _log_cal("off", p_off_cal, y_off)
        _log_cal("any", p_any_cal, y_any)
        # ------------------------------------------------------



        # ............................................................
        # SAVE-BEST BY EXTRINSIC: in pretrain, env returns 0 reward but
        # writes per-step extrinsic to 'ext_reward_step'. Fallback to
        # 'reward' if that key doesn't exist.
        if len(replay) < max(args.batch_steps, args.train_fill):
            pass
        else:
            nonlocal best_ext_return  # so we can update the outer variable
            if "ext_reward_step" in ep_info:
                ext_return = float(np.asarray(ep_info["ext_reward_step"], np.float64).sum())
            else:
                ext_return = float(np.asarray(ep["reward"], np.float64).sum())

            if ext_return > best_ext_return:
                best_ext_return = ext_return
                best_ckpt.save()
                logger.add({"best_ext_return": best_ext_return, "best_step": int(step)},
                        prefix="checkpoint")
                print(f"[Checkpoint] New BEST extrinsic return {best_ext_return:.2f} at step {int(step)} → saved checkpoint_best.ckpt")
        # ............................................................



        print(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = ep[key].max(0).mean()
        metrics.add(stats, prefix="stats")

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, ep_info, worker: per_episode(ep,ep_info))
    driver.on_episode(lambda ep, ep_info, worker: _csv_log(ep, ep_info))
    # driver.on_episode(lambda ep, ep_info, worker: _csv_log(ep, ep_info, is_eval=False))
    driver.on_step(lambda _, __, ___: step.increment())
    
    def penalise_and_store(tran, _, worker):
        # Ensure action has consistent 1D shape across steps (e.g., (54,)).
        if "action" in tran:
            a = np.asarray(tran["action"])
            if a.ndim >= 2:
                # Remove leading singleton batch dims, then flatten any 2D one-hot grids.
                while a.ndim > 1 and a.shape[0] == 1:
                    a = a[0]
                if a.ndim > 1:
                    a = a.reshape(-1)
            tran["action"] = a.astype(np.float32, copy=False)
        # Ensure unsafe flag has consistent shape (1,) across steps.
        unsafe = np.asarray(tran.get("unsafe", 0.0), dtype=np.float32)
        if unsafe.ndim == 0:
            unsafe = unsafe.reshape(1)
        else:
            # If array of any length, take first element to get shape (1,)
            unsafe = unsafe.reshape(-1)[:1]
        tran["unsafe"] = unsafe

        if "unsafe" not in tran:
            tran["unsafe"] = np.array([0.0], np.float32)
        if np.any(tran["unsafe"]):
            tran["reward"] -= getattr(args, "shield_penalty", 0.1)

        # human-readable log per step (Python print; low overhead with B=1)
        if float(tran["unsafe"][0]) > 0.5:
            idx = int(np.argmax(tran["action"]))          # one-hot -> index after override
            # 'step' is the embodied.Counter from outer scope.
            print(f"[Shield] step={int(step)} override -> idx={idx}")


        replay.add(tran, worker)

        # Optional per-step CSV logging for training debugging/analysis.
        if getattr(args, "log_per_step", False):
            try:
                row = {
                    "step": int(step),
                    "env_step": int(np.asarray(tran.get("env_step", -1)).reshape(-1)[0]) if "env_step" in tran else -1,
                    "reward": float(np.asarray(tran.get("reward", 0.0)).reshape(-1)[0]) if "reward" in tran else 0.0,
                    "unsafe": float(np.asarray(tran.get("unsafe", 0.0)).reshape(-1)[0]) if "unsafe" in tran else 0.0,
                    "action": tran.get("action").tolist() if hasattr(tran.get("action"), "tolist") else str(tran.get("action")),

                    "speed_ms": float(np.asarray(tran.get("speed_ms", 0.0)).reshape(-1)[0]) if "speed_ms" in tran else 0.0,
                    "comfort_acc_ms2": float(np.asarray(tran.get("comfort_acc_ms2", 0.0)).reshape(-1)[0]) if "comfort_acc_ms2" in tran else 0.0,
                    "comfort_jerk_ms3": float(np.asarray(tran.get("comfort_jerk_ms3", 0.0)).reshape(-1)[0]) if "comfort_jerk_ms3" in tran else 0.0,
                    "comfort_dsteer_abs": float(np.asarray(tran.get("comfort_dsteer_abs", 0.0)).reshape(-1)[0]) if "comfort_dsteer_abs" in tran else 0.0,
                    "comfort_dthrottle_abs": float(np.asarray(tran.get("comfort_dthrottle_abs", 0.0)).reshape(-1)[0]) if "comfort_dthrottle_abs" in tran else 0.0,
                    "comfort_lat_acc_ms2": float(np.asarray(tran.get("comfort_lat_acc_ms2", 0.0)).reshape(-1)[0]) if "comfort_lat_acc_ms2" in tran else 0.0,
                    "traffic_density": int(np.asarray(tran.get("traffic_density", -1)).reshape(-1)[0]) if "traffic_density" in tran else -1,
                }
                train_csv_writer.writerow(row)
                train_csv_file.flush()
            except Exception:
                pass

    driver.on_step(penalise_and_store)

    def tap_shield(tran, _, worker):
        if "log_shield_unsafe" in tran:
            sb.set(sb.ShieldInfo(
                unsafe     = int(tran["log_shield_unsafe"][0]),
                orig_idx   = int(tran["log_shield_orig_idx"]),
                idx        = int(tran["log_shield_idx"]),
                orig_risk  = float(tran["log_shield_orig_risk"]),
                chosen_risk= float(tran["log_shield_chosen_risk"]),
                lam         = float(np.asarray(tran["log_shield_lam"]).reshape(-1)[0]),
                # min_left   = float(tran["log_shield_min_left"]),
                # min_right  = float(tran["log_shield_min_right"]),
            ))
    driver.on_step(tap_shield)

    print("Prefill train dataset.")
    random_agent = embodied.RandomAgent(env.act_space, args.actor_dist_disc)
    while len(replay) < max(args.batch_steps, args.train_fill):
        driver(random_agent.policy, steps=100)
    logger.add(metrics.result())

    logger.write()

    dataset = agent.dataset(replay.dataset)
    state = [None]  # To be writable from train step function below.
    batch = [None]

    def train_step(_, __, ___):
        for _ in range(should_train(step)):
            with timer.scope("dataset"):
                batch[0] = next(dataset)
            outs, state[0], mets = agent.train(batch[0], state[0])
            metrics.add(mets, prefix="train")

            if getattr(replay, "update_visit_count", False):
                replay.update_visit_count(jax.device_get(batch[0]["env_step"]))

            if "key" in outs:
                replay.prioritize(outs["key"], outs["env_step"], outs["model_loss"], outs["td_error"])

            updates.increment()
        if should_sync(updates):
            agent.sync()
        if should_log(step):
            agg = metrics.result()
            report = agent.report(batch[0])
            report = {k: v for k, v in report.items() if "train/" + k not in agg}
            logger.add(agg)
            logger.add(report, prefix="report")
            logger.add(replay.stats, prefix="replay")
            logger.add(timer.stats(), prefix="timer")
            logger.add({"lam": float(lam), "cost_ema": float(cost_ema)}, prefix="shield")

            # logger.add({"shield/override_rate": shield.override_rate})
            logger.write(fps=True)

    driver.on_step(train_step)

    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    timer.wrap("checkpoint", checkpoint, ["save", "load"])
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    checkpoint.load_or_save()
    should_save(step)  # Register that we jused saved.

    # === Best-by-extrinsic checkpoint ===
    # best_ckpt = embodied.Checkpoint(logdir / "checkpoint_best.ckpt")
    # timer.wrap("checkpoint", checkpoint, ["save", "load"])
    # best_ckpt.step = step
    # best_ckpt.agent = agent    # you can omit replay to keep it small
    # best_ckpt.replay = replay
    # best_ext_return = -np.inf  # running best extrinsic episode return

    print("Start training loop.")
    driver._state = None
    print("*********************************************************")
    if should_expl(step):
        print("Exploration mode: using exploration policy.")
    print("*********************************************************")

    # lam = 0.0
    # lam_dev = jnp.array(lam, dtype=jnp.float32)
    # eps = getattr(args, "cost_budget", 0.02)     # target per-step cost rate
    # beta = getattr(args, "dual_lr", 1e-2)        # dual step size
    # c_ma = 0.0                                    # EMA of observed cost

    # def _dual_update(tran, _, __):
    #     nonlocal lam, lam_dev, c_ma
    #     def _get01(x):
    #         if x is None: return 0.0
    #         v = float(np.asarray(x).reshape(-1)[0])
    #         return 1.0 if v > 0.0 else 0.0

    #     # Use realized costs, not predictions
    #     c_col = _get01(tran.get("collision"))
    #     c_off = _get01(tran.get("offlane")) or _get01(tran.get("lane_invasion"))
    #     c = 1.0 if (c_col or c_off) else 0.0

    #     # Smooth and update lambda
    #     c_ma = 0.99 * c_ma + 0.01 * c
    #     lam = max(0.0, lam + beta * (c_ma - eps))
    #     lam_dev = jnp.array(lam, dtype=jnp.float32)

    # driver.on_step(_dual_update)

    # lam = 0.0
    # cost_ema = 0.0
    # ema = 0.01

    # def dual_step(tran, _, __):
    #     nonlocal lam, cost_ema
    #     as01 = lambda x: float(np.asarray(x).reshape(-1)[0] > 0) if x is not None else 0.0
    #     c = 1.0 if (as01(tran.get("collision")) or as01(tran.get("offlane")) or as01(tran.get("lane_invasion"))) else 0.0
    #     cost_ema = (1 - ema) * cost_ema + ema * c
    #     lam = max(0.0, lam + args.dual_lr * (cost_ema - args.cost_budget))

    # driver.on_step(dual_step)

    # logger.add({"lam": float(lam)}, prefix="shield")
    # # logger.add({"lam": float(lam), "cost_ma": float(c_ma)}, prefix="shield")

    # # policy = lambda *args: agent.policy(*args, mode="explore" if should_expl(step) else "train")

    # shield_warmup = getattr(args, "shield_warmup_steps", 5_000)

    # def policy_with_lam(obs, state, **_):
    #     warm = (int(step) < shield_warmup)
    #     return agent.policy(
    #         obs, state, mode=("explore" if should_expl(step) else "train"),
    #         lam=jax.device_put(lam), warmup=warm
    #     )
    # policy = policy_with_lam

    # --- top-level defaults
    cost_budget = 0.03
    dual_lr     = 1e-2
    # shield_warmup    = getattr(args, "shield_warmup_steps", 5_000)

    lam = 0.0
    cost_ema, ema = 0.0, 0.01

    # print(1)

    def dual_step(tran, _, __):
        nonlocal lam, cost_ema
        as01 = lambda x: float(np.asarray(x).reshape(-1)[0] > 0) if x is not None else 0.0
        c = 1.0 if (as01(tran.get("collision")) or as01(tran.get("offlane")) or as01(tran.get("lane_invasion"))) else 0.0
        cost_ema = (1 - ema) * cost_ema + ema * c
        lam = max(0.0, lam + dual_lr * (cost_ema - cost_budget))
    # print(2)
    # driver.on_step(dual_step)

    shield_warmup = 5_000
    lam_fixed_max = 0.15  # small but effective

    def policy_with_lam(obs, state, **_):
        s = int(step)

        lam_val = 0.0 if s < shield_warmup else lam_fixed_max * min(1.0, (s - shield_warmup) / 25_000)

        B = int(np.asarray(obs["is_first"]).shape[0])
        obs = dict(obs)
        obs["log_shield_lam"]     = np.full((B,), lam_val, np.float32)
        obs["log_shield_warmup"]  = np.full((B,), 1.0 if s < shield_warmup else 0.0, np.float32)
        # print("---------------------------------------------------------")
        return agent.policy(obs, state, mode=("explore" if should_expl(step) else "train"))

    policy = policy_with_lam



    # # Evaluate shield under ninjax context to avoid "Wrap impure functions in pure()" errors.
    # _pure_shield = nj.pure(lambda lat, act: shield.apply(lat, act), nested=True)

    # def shielded_policy(obs, state):
    #     outs, new_state = agent.policy(obs, state, mode="explore" if should_expl(step) else "train")
    #     latent = new_state[0][0]
    #     (safe_act, unsafe), _ = _pure_shield(
    #         agent.varibs,
    #         jax.random.PRNGKey(0),
    #         latent,
    #         outs["action"],
    #         create=False,
    #         modify=False,
    #         ignore=True,
    #     )
    #     outs["action"] = safe_act
    #     outs["unsafe"] = unsafe.astype(jnp.float32)
    #     return outs, new_state

    # policy = shielded_policy

    while step < args.steps:
        driver(policy, steps=100)
        if should_save(step):
            checkpoint.save()
    logger.write()
    # cleanup csv file if opened
    try:
        if train_csv_file is not None:
            train_csv_file.close()
    except Exception:
        pass
