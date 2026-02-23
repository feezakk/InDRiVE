import datetime
import os
import warnings

import embodied
import ruamel.yaml as yaml

import car_dreamer
import dreamerv3

import numpy as np
# from safety import LongHorizonShield, ActionFilterShield, fallback_subset_from_cfg

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
    # print(config)

    # Log active exploration and reward settings early so runs are obvious in terminal
    dreamerv3_config = config.dreamerv3
    try:
        expl_rewards = dict(dreamerv3_config.expl_rewards)
    except Exception:
        expl_rewards = getattr(dreamerv3_config, "expl_rewards", {})
    print("[startup] expl_behavior:", getattr(dreamerv3_config, "expl_behavior", None))
    print("[startup] expl_rewards:", expl_rewards)

    # Defensive check: if this looks like a pretrain run (logdir contains 'pretrain' or
    # PRETRAIN env var is set), ensure extrinsic reward scale is zero to avoid accidental
    # extrinsic training. Fail fast with a clear message.
    is_pretrain = "pretrain" in str(dreamerv3_config.logdir).lower() or bool(int(os.environ.get("PRETRAIN", "0")))
    if is_pretrain:
        extr = expl_rewards.get("extr", 0.0) if isinstance(expl_rewards, dict) else 0.0
        if float(extr) != 0.0:
            raise RuntimeError(
                f"Refusing to start pretrain run: detected non-zero extrinsic reward scale (expl_rewards.extr={extr}).\n"
                "Set --dreamerv3.expl_rewards.extr 0.0 for intrinsic-only pretraining."
            )

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

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_filename = f"config_{timestamp}.yaml"
    config.save(str(logdir / config_filename))
    print(f"[Train] Config saved to {logdir / config_filename}")

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, dreamerv3_config)
    # print(agent.__dict__.keys())
    # Create safety shield only if enabled in config. This prevents the shield from
    # being active during pretraining when `--dreamerv3.safe_train.enable False` is passed.
    if getattr(dreamerv3_config, "safe_train", {}).get("enable", False):
        # Default to one-step action filter if enabled; other shield types may be
        # constructed elsewhere based on config.safe_train.mode.
        shield = dreamerv3.ActionFilterShield(agent.agent.wm, env.act_space["action"], gamma=0.99)
    else:
        shield = None
    replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "replay")
    args = embodied.Config(
        **dreamerv3_config.run,
        logdir=dreamerv3_config.logdir,
        batch_steps=dreamerv3_config.batch_size * dreamerv3_config.batch_length,
        actor_dist_disc=dreamerv3_config.actor_dist_disc,
    )
    embodied.run.train(agent, env, replay, shield, logger, args)


if __name__ == "__main__":
    main()
