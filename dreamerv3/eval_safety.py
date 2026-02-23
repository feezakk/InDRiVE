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



# def eval_safety(agent, env, logger, args):
def eval_safety(agent, env, logger, args, safe_eval_cfg=None):
    print("Start evaluation.")
    logdir = embodied.Path(args.logdir); logdir.mkdirs()
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

    train_csv_f, train_csv_w = _make_writer(logdir / "train_success.csv")
    eval_csv_f,  eval_csv_w  = _make_writer(logdir / "eval_success.csv")
    atexit.register(lambda: (train_csv_f.close(), eval_csv_f.close()))
    train_ep_idx = {"v": 0}
    eval_ep_idx  = {"v": 0}

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
            "episode_index": (eval_ep_idx["v"] if is_eval else train_ep_idx["v"]),
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
        w, f = (eval_csv_w, eval_csv_f) if is_eval else (train_csv_w, train_csv_f)
        w.writerow(row); f.flush()
        if is_eval: eval_ep_idx["v"] += 1
        else:       train_ep_idx["v"] += 1
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
    
    eval_metrics = EvalMetrics(
        outdir=str(embodied.Path(args.logdir)),
        fps=fps, agg=agg, tau=tau, calibrator_path=calpath
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
    while step < args.steps:
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
            "run.steps": 5e4,
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

