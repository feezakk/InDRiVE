"""
Run a single eval episode using a saved checkpoint and log per-step info to CSV.
Usage:
    python tools/log_eval_episode.py --checkpoint ./logdir/your_run/checkpoint.ckpt --port 2002 --out eval_episode.csv

This script uses the existing task factory in car_dreamer to create the env and loads the dreamerv3 Agent.
"""
import argparse
import csv
import os
import time

import embodied
import ruamel.yaml as yaml

import car_dreamer
import dreamerv3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--port", type=int, default=2002)
    p.add_argument("--task", type=str, default="carla_lane_following")
    p.add_argument("--out", type=str, default="eval_episode.csv")
    p.add_argument("--render", action="store_true")
    args = p.parse_args()

    model_configs = yaml.YAML(typ="safe").load((embodied.Path(__file__).parent.parent / "dreamerv3" / "dreamerv3.yaml").read())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    config = config.update({"dreamerv3": model_configs["small"]})

    parsed, other = embodied.Flags(task=[args.task]).parse_known(['--task', args.task, f'--env.world.carla_port', str(args.port)])
    for name in parsed.task:
        env, env_config = car_dreamer.create_task(name, ['--task', name, f'--env.world.carla_port', str(args.port)])
        config = config.update(env_config)
    config = embodied.Flags(config).parse(other)

    dreamerv3_config = config.dreamerv3
    from embodied.envs import from_gym
    env = from_gym.FromGym(env)
    env = dreamerv3.train.wrap_env(env, dreamerv3_config) if hasattr(dreamerv3, 'train') else env
    env = embodied.BatchEnv([env], parallel=False)

    step = embodied.Counter()
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, dreamerv3_config)

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load(args.checkpoint, keys=["agent"])

    obs = env.reset()
    state = agent.policy_initial(1)

    rows = []

    done = False
    total_reward = 0.0
    while not done:
        outs, state = agent.policy(obs, state, mode="eval")
        action = outs.get("action")
        obs, rew, done, info = env.step(action)
        speed_kmh = info.get("speed_kmh", None)
        shield_unsafe = info.get("log_shield_unsafe", None) or info.get("eval_log_shield_unsafe", None)
        rows.append({
            "time_step": info.get("time_steps", 0),
            "speed_kmh": speed_kmh,
            "action": action.tolist() if hasattr(action, 'tolist') else action,
            "reward": float(rew),
            "shield_unsafe": shield_unsafe,
            **{k: info.get(k) for k in ("r_speed", "r_collision", "r_out_of_lane")} }
        )
        total_reward += float(rew)
        if args.render:
            time.sleep(0.01)

    # write CSV
    keys = rows[0].keys() if rows else ["time_step","speed_kmh","action","reward","shield_unsafe"]
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Episode finished. total_reward={total_reward:.2f}. CSV written to {args.out}")


if __name__ == '__main__':
    main()
