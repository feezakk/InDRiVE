import re
import embodied
import numpy as np
from dreamerv3 import shield_bus as sb

import ruamel.yaml as yaml
import warnings

import car_dreamer
import dreamerv3
from tools.eval_metrics import EvalMetrics
import csv, atexit

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


def wrap_env(env, config):
    args = config.wrapper
    env = embodied.wrappers.InfoWrapper(env)
    for name, space in env.act_space.items():
        if name == "reset":
            continue
        elif space.discrete:
            env = embodied.wrappers.OneHotAction(env, name)
        elif args.discretize:
            env = embodied.wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.ExpandScalars(env)
    if args.length:
        env = embodied.wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env

import os

# def eval_safety(agent, env, logger, args):
def eval_safety(agent, env, logger, args, safe_eval_cfg=None):
    print("Start evaluation.")
    logdir = embodied.Path(args.logdir); logdir.mkdirs()
    import csv, atexit

    EP_FIELDS = [
        "episode_index", "env_step", "length", "return",
        "success",
        "collision", "off_road", "out_of_lane", "wrong_direction", "too_slow", "time_exceeded", "destination_reached",
        "mean_speed_ms", "std_speed_ms",
        "mean_abs_acc_ms2", "mean_abs_jerk_ms3",
        "mean_abs_dsteer", "mean_abs_dthrottle",
        "mean_abs_lat_acc_ms2",
        "lane_pair_index", "traffic_density",
    ]
    ep_csv_path = str(logdir / "eval_episode_metrics.csv")
    ep_csv_exists = embodied.Path(ep_csv_path).exists()
    ep_csv_f = open(ep_csv_path, "a", newline="")
    ep_csv_w = csv.DictWriter(ep_csv_f, fieldnames=EP_FIELDS)
    if not ep_csv_exists:
        ep_csv_w.writeheader()
        ep_csv_f.flush()
    atexit.register(lambda: ep_csv_f.close())
    ep_ep_idx = {"v": 0}

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
                return np.asarray(ep[k]).reshape(-1)[:n].astype(np.float32)
            if k in ep_info:
                return np.asarray(ep_info[k]).reshape(-1)[:n].astype(np.float32)
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
    step = logger.step
    agg = embodied.Metrics()
    print("Observation space:", env.obs_space)
    print("Action space:", env.act_space)

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy"])
    timer.wrap("env", env, ["step"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    # ---------- CSV writers (NEW) ----------
    def _make_writer(path):
        path = embodied.Path(path)
        exists = path.exists()
        f = open(str(path), "a", newline="")
        fieldnames = [
            "episode_index","env_step","length","return",
            "is_collision", "out_of_lane", "destination_reached", "wrong_direction", "too_slow", "off_road" ,
            "lat_mean","lat_max"  
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader(); f.flush()
        return f, w

    train_succ_f, train_succ_w = _make_writer(logdir / "train_success.csv")
    eval_succ_f,  eval_succ_w  = _make_writer(logdir / "eval_success.csv")
    atexit.register(lambda: (train_succ_f.close(), eval_succ_f.close()))
    train_succ_idx = {"v": 0}
    eval_succ_idx  = {"v": 0}

    def _csv_log(ep, ep_info, is_eval=False):
        # Episode stats
        length = int(len(ep["reward"]) - 1)
        ret = float(ep["reward"].astype(np.float64).sum())
        def _any(k):  # handles missing keys
            v = ep_info.get(k, [])
            return bool(np.any(np.array(v)))
        

        lat = np.asarray(ep_info.get("lat_err", []), dtype=np.float64).reshape(-1)
        # ignore NaNs
        m = np.isfinite(lat)
        lat_mean = float(np.nan) if not m.any() else float(lat[m].mean())
        lat_max  = float(np.nan) if not m.any() else float(np.abs(lat[m]).max())  # or plain max()
        row = {
            "episode_index": (eval_succ_idx["v"] if is_eval else train_succ_idx["v"]),
            "env_step": int(logger.step),
            "length": length,
            "return": ret,
            "is_collision": int(_any("is_collision")),
            "out_of_lane": int(_any("out_of_lane")),
            "destination_reached": int(_any("destination_reached")),
            "wrong_direction": int(_any("wrong_direction")),
            "too_slow": int(_any("too_slow")),
            "off_road": int(_any("off_road")),
            "lat_mean": lat_mean,
            "lat_max": lat_max,
        }
        w, f = (eval_succ_w, eval_succ_f) if is_eval else (train_succ_w, train_succ_f)
        w.writerow(row); f.flush()
        if is_eval: eval_succ_idx["v"] += 1
        else:       train_succ_idx["v"] += 1
    # ---------------------------------------


    def per_episode(ep, ep_info, _worker=None):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        success = float(np.any(np.array(ep_info.get("goal_reached", [0]))))
        collision = float(np.any(np.array(ep_info.get("collision", [0]))))
        past_goal = float(np.any(np.array(ep_info.get("past_goal", [0]))))
        not_moving = float(np.any(np.array(ep_info.get("not_moving", [0]))))
        time_exceeded = float(np.any(np.array(ep_info.get("time_exceeded", [0]))))

        logger.add({"length": length, 
                    "score": score, 
                    "success": success,
                    "collision": collision,
                    "past_goal": past_goal,
                    "not_moving": not_moving,
                    "time_exceeded": time_exceeded
                    }, prefix="episode")
        print(f"Episode has {length} steps and return {score:.1f}.")

        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]

        def log(k, v):
            if re.match(args.log_keys_sum,  k): stats[f"sum_{k}"]  = v.sum()
            if re.match(args.log_keys_mean, k): stats[f"mean_{k}"] = v.mean()
            if re.match(args.log_keys_max,  k): stats[f"max_{k}"]  = v.max(0).mean()

        for k, v in ep.items():
            if not args.log_zeros and k not in nonzeros and (v == 0).all():
                continue
            nonzeros.add(k); log(k, v)
        for k, v in ep_info.items():
            log(k, v)

        length = int(len(ep["reward"]) - 1)
        ret = float(ep["reward"].astype(np.float64).sum())
        n = max(1, length)

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

        speed_ms = _series(ep, ep_info, ["speed_ms", "speed_norm"], n, default=0.0)
        acc_ms2 = _series(ep, ep_info, ["comfort_acc_ms2"], n, default=0.0)
        jerk_ms3 = _series(ep, ep_info, ["comfort_jerk_ms3"], n, default=0.0)
        dsteer = _series(ep, ep_info, ["comfort_dsteer_abs"], n, default=0.0)
        dthr = _series(ep, ep_info, ["comfort_dthrottle_abs"], n, default=0.0)
        latacc = _series(ep, ep_info, ["comfort_lat_acc_ms2"], n, default=0.0)

        # lane_pair = ep_info.get("lane_pair_index", ep.get("lane_pair_index", "NA"))
        lp_series = _series(ep, ep_info, ["lane_pair_index"], n, default=-1)
        lane_pair = int(lp_series[0]) if lp_series.size else -1
        # dens = ep_info.get("traffic_density", ep.get("traffic_density", "NA"))
        dens_series = _series(ep, ep_info, ["traffic_density"], n, default=-1)
        dens = int(dens_series[0]) if dens_series.size else -1

        row = {
            "episode_index": int(eval_succ_idx["v"]),
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

            # "lane_pair_index": lane_pair if isinstance(lane_pair, (int, float, str)) else "NA",
            # "traffic_density": dens if isinstance(dens, (int, float, str)) else "NA",
            "lane_pair_index": int(lane_pair),
            "traffic_density": int(dens),
        }
        ep_csv_w.writerow(row)
        ep_csv_f.flush()
        ep_ep_idx["v"] += 1

        logger.add(agg.result()); logger.add(timer.stats(), prefix="timer"); logger.write(fps=True)
        agg.add(stats, prefix="stats")

    def per_step(tran, _info, _worker=None):
        # Push shield diagnostics to HUD via bus, if present.
        if "log_shield_unsafe" in tran:
            sb.set(sb.ShieldInfo(
                unsafe      = int(np.asarray(tran["log_shield_unsafe"]).reshape(-1)[0] > 0.5),
                orig_idx    = int(np.asarray(tran["log_shield_orig_idx"]).reshape(-1)[0]),
                idx         = int(np.asarray(tran["log_shield_idx"]).reshape(-1)[0]),
                orig_risk   = float(np.asarray(tran["log_shield_orig_risk"]).reshape(-1)[0]),
                chosen_risk = float(np.asarray(tran["log_shield_chosen_risk"]).reshape(-1)[0]),
                lam         = float(np.asarray(tran["log_shield_lam"]).reshape(-1)[0]),
            ))
        step.increment()

    driver = embodied.Driver(env)

    # --- attach EvalMetrics (before registering other callbacks) ---
    from tools.eval_metrics import EvalMetrics
    # BatchEnv usually has no .fps; fall back to 10 Hz.
    fps = getattr(env, "fps", 10.0)
    tau = getattr(safe_eval_cfg, "tau", 0.3) if safe_eval_cfg else 0.3
    agg_mode = getattr(safe_eval_cfg, "agg", "max") if safe_eval_cfg else "max"
    calpath = getattr(safe_eval_cfg, "calibrator_path", None) if safe_eval_cfg else None

    metrics = EvalMetrics(outdir=args.logdir, fps=fps, agg=agg_mode, tau=tau,
                        calibrator_path=calpath)
    
    # eval_metrics = EvalMetrics(
    #     outdir=str(embodied.Path(args.logdir)),
    #     fps=fps, agg=agg, tau=tau, calibrator_path=calpath
    # )
    eval_metrics = EvalMetrics(
        outdir=str(embodied.Path(args.logdir)),
        fps=fps, agg=agg_mode, tau=tau, calibrator_path=calpath
    )
    driver.on_episode(lambda ep, ep_info, worker: eval_metrics.on_episode(ep, ep_info))
    driver.on_episode(lambda ep, ep_info, worker: _csv_log(ep, ep_info, is_eval=True))

    # ---------------------------------------

    driver.on_episode(per_episode)
    driver.on_step(per_step)

    # driver.on_episode(per_episode)
    # driver.on_step(per_step)

    checkpoint = embodied.Checkpoint();
    checkpoint.agent = agent
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint, keys=["agent"])
    else:
        raise ValueError("No checkpoint specified.")

    print("Start evaluation loop.")
    policy = lambda *x: agent.policy(*x, mode="eval")
    eval_episodes = int(os.environ.get("EVAL_EPISODES", "50"))
    # while step < args.steps:
    #     driver(policy, steps=100)
    while (ep_ep_idx["v"] < eval_episodes) and (step < args.steps):
        driver(policy, steps=100)
    logger.write()
    eval_metrics.close()

def main(argv=None):
    model_configs = yaml.YAML(typ="safe").load((embodied.Path(__file__).parent / "dreamerv3.yaml").read())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    config = config.update({"dreamerv3": model_configs["small"]})

    parsed, other = embodied.Flags(task=["carla_navigation"]).parse_known(argv)
    for name in parsed.task:
        print("Using task: ", name)
        env, env_config = car_dreamer.create_task(name, argv)
        config = config.update(env_config)
    config = embodied.Flags(config).parse(other)

    logdir = embodied.Path(config.dreamerv3.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
        ],
    )

    from embodied.envs import from_gym

    dreamerv3_config = config.dreamerv3
    env = from_gym.FromGym(env)
    env = wrap_env(env, dreamerv3_config)
    env = embodied.BatchEnv([env], parallel=False)

    dreamerv3_config = dreamerv3_config.update(
        {
            "run.log_keys_sum": "(travel_distance|destination_reached|out_of_lane|time_exceeded|is_collision|timesteps)",
            "run.log_keys_mean": "(travel_distance|ttc|speed_norm|wpt_dis)",
            "run.log_keys_max": "(travel_distance|ttc|speed_norm|wpt_dis)",
        }
    )

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, dreamerv3_config)
    args = embodied.Config(
        **dreamerv3_config.run,
        logdir=dreamerv3_config.logdir,
        batch_steps=dreamerv3_config.batch_size * dreamerv3_config.batch_length,
    )



    safe_eval_cfg = getattr(dreamerv3_config, "safe_eval", None)
    eval_safety(agent, env, logger, args, safe_eval_cfg)
    # eval_safety(agent, env, logger, args)


if __name__ == "__main__":
    main()

