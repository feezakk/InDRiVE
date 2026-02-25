#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any

def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _summarize_episode_csv(rows: List[Dict[str, str]]) -> Dict[str, float]:
    # expects columns: success, collision, off_road, out_of_lane, wrong_direction, too_slow, time_exceeded, destination_reached
    if not rows:
        return {"episodes": 0, "sr": 0.0}
    n = len(rows)
    sr = sum(int(_safe_float(r.get("success", 0))) for r in rows) / n

    def mean(key: str) -> float:
        return sum(_safe_float(r.get(key, 0.0)) for r in rows) / n

    out = {
        "episodes": float(n),
        "sr": 100.0 * sr,
        "collision_rate": 100.0 * mean("collision"),
        "off_road_rate": 100.0 * mean("off_road"),
        "out_of_lane_rate": 100.0 * mean("out_of_lane"),
        "wrong_direction_rate": 100.0 * mean("wrong_direction"),
        "too_slow_rate": 100.0 * mean("too_slow"),
        "time_exceeded_rate": 100.0 * mean("time_exceeded"),

        # comfort summaries if present
        "mean_abs_acc_ms2": mean("mean_abs_acc_ms2"),
        "mean_abs_jerk_ms3": mean("mean_abs_jerk_ms3"),
        "mean_abs_dsteer": mean("mean_abs_dsteer"),
        "mean_abs_dthrottle": mean("mean_abs_dthrottle"),
        "mean_abs_lat_acc_ms2": mean("mean_abs_lat_acc_ms2"),
        "mean_speed_ms": mean("mean_speed_ms"),
        "std_speed_ms": mean("std_speed_ms"),
    }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carla_port", type=int, required=True)
    ap.add_argument("--gpu", type=str, required=True)
    ap.add_argument("--eval_sh", type=str, required=True, help="Path to your eval bash wrapper")
    ap.add_argument("--routes_json", type=str, required=True, help="Route groups JSON (straight/turn/multi_turn)")
    ap.add_argument("--spec_json", type=str, required=True, help="Sweep spec JSON (methods/phases/seeds/checkpoints)")
    ap.add_argument("--out_root", type=str, required=True)
    args, unknown = ap.parse_known_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    routes = json.loads(Path(args.routes_json).read_text())
    spec = json.loads(Path(args.spec_json).read_text())
    seeds = spec["seeds"]
    phases = spec["phases"]
    methods = spec["methods"]
    eval_steps = int(spec.get("eval", {}).get("steps", 50000))
    eval_episodes = int(spec.get("eval", {}).get("episodes", 50))
    eval_densities = spec.get("eval", {}).get("densities", [5, 10, 20])
    eval_tasks = spec.get("eval", {}).get("tasks", [])
    if not eval_tasks:
        raise ValueError("spec_json eval.tasks is required")

    # merged outputs
    merged_episode_csv = out_root / "eval_episodes_all.csv"
    summary_csv = out_root / "eval_summary.csv"

    merged_fields_written = merged_episode_csv.exists()
    summary_fields_written = summary_csv.exists()

    # We will append to merged CSV and summary CSV
    merged_f = merged_episode_csv.open("a", newline="")
    summary_f = summary_csv.open("a", newline="")

    merged_writer = None
    summary_writer = None

    try:
        import os

        import statistics
        from collections import defaultdict

        all_summary_rows = []

        for method_name, ckpt_map in methods.items():
            for phase in phases:
                if phase not in ckpt_map:
                    continue
                ckpt_tmpl = ckpt_map[phase]
                if not ckpt_tmpl:
                    continue

                # stage used by RoutePoolMixin (must exist)
                stage = "zero_shot" if phase == "zero_shot" else "eval"

                for task in eval_tasks:
                    if task not in routes:
                        raise KeyError(f"routes_json missing task '{task}'")
                    task_routes = routes[task]

                    # for dens in eval_densities:
                    #     for route_type, route_indices in task_routes.items():
                    #         for route_idx in route_indices:
                    #             for seed in seeds:
                    #                 ckpt = ckpt_tmpl.format(seed=seed)
                    #                 ckpt_path = Path(ckpt)
                    #                 if not ckpt_path.exists():
                    #                     raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

                    #                 run_name = f"{task}__{method_name}__{phase}__dens{dens}__{route_type}__r{route_idx}__seed{seed}"
                    #                 logdir = out_root / "runs" / run_name
                    #                 logdir.mkdir(parents=True, exist_ok=True)

                    #                 cmd = [
                    #                     str(Path(args.eval_sh).resolve()),
                    #                     str(args.carla_port),
                    #                     str(args.gpu),
                    #                     str(ckpt_path.resolve()),
                    #                     run_name,
                    #                     "--task", task,
                    #                     "--dreamerv3.logdir", str(logdir),
                                        
                    #                     "--dreamerv3.run.steps", str(eval_steps),
                    #                     "--env.stage", stage,
                    #                     "--env.lane_pair_index", str(route_idx),
                    #                     "--env.num_vehicles", str(dens),
                    #                 ] + unknown

                    for route_type in task_routes.keys():   # now should be: straight, right, left, multi_turn
                        for seed in seeds:
                            ckpt = ckpt_tmpl.format(seed=seed)
                            ckpt_path = Path(ckpt)
                            if not ckpt_path.exists():
                                raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
                            run_name = f"{task}__{method_name}__{phase}__{route_type}__seed{seed}"
                            logdir = out_root / "runs" / run_name

                            cmd = [
                                str(Path(args.eval_sh).resolve()),
                                str(args.carla_port),
                                str(args.gpu),
                                str(ckpt_path.resolve()),
                                run_name,
                                "--task", task,
                                "--dreamerv3.logdir", str(logdir),
                                "--dreamerv3.run.steps", str(eval_steps),
                                "--env.stage", stage,
                                "--env.route_group", str(route_type),   # <<<<< this is the new control knob
                            ] + unknown

                            print("[RUN]", " ".join(cmd), flush=True)

                            env_vars = dict(os.environ)
                            env_vars["EVAL_EPISODES"] = str(eval_episodes)
                            subprocess.run(cmd, check=True, env=env_vars)



                            # Consume per-episode CSV produced by eval script
                            ep_csv = logdir / "eval_episode_metrics.csv"
                            if not ep_csv.exists():
                                raise FileNotFoundError(
                                    f"Expected {ep_csv} to exist. "
                                    f"Your eval code must write eval_episode_metrics.csv."
                                )

                            ep_rows = _read_csv_rows(ep_csv)

                            # Write merged per-episode file with metadata columns
                            for r in ep_rows:
                                r2 = dict(r)
                                r2.update({
                                    "method": method_name,
                                    "phase": phase,
                                    "route_type": route_type,
                                    # "route_idx": str(route_idx),
                                    "seed": str(seed),
                                    "checkpoint": str(ckpt_path),
                                    "logdir": str(logdir),
                                    "task": task,
                                    # "traffic_density": str(dens),
                                })
                                if merged_writer is None:
                                    merged_writer = csv.DictWriter(merged_f, fieldnames=list(r2.keys()))
                                    if not merged_fields_written:
                                        merged_writer.writeheader()
                                        merged_fields_written = True
                                merged_writer.writerow(r2)
                            merged_f.flush()

                            # Write per-run summary row
                            summ = _summarize_episode_csv(ep_rows)
                            summ_row = {
                                "method": method_name,
                                "phase": phase,
                                "route_type": route_type,
                                # "route_idx": int(route_idx),
                                "seed": int(seed),
                                "checkpoint": str(ckpt_path),
                                "logdir": str(logdir),
                                "task": task,
                                # "traffic_density": int(dens),
                                **summ,
                            }
                            all_summary_rows.append(summ_row)
                            if summary_writer is None:
                                summary_writer = csv.DictWriter(summary_f, fieldnames=list(summ_row.keys()))
                                if not summary_fields_written:
                                    summary_writer.writeheader()
                                    summary_fields_written = True
                            summary_writer.writerow(summ_row)
                            summary_f.flush()

        print(f"[DONE] Wrote: {merged_episode_csv} and {summary_csv}")

        group_keys = ["method", "phase", "task", "route_type", "route_idx", "traffic_density"]

        metric_keys = [
            "sr",
            "collision_rate", "off_road_rate", "out_of_lane_rate",
            "wrong_direction_rate", "too_slow_rate", "time_exceeded_rate",
            "mean_abs_acc_ms2", "mean_abs_jerk_ms3", "mean_abs_lat_acc_ms2",
            "mean_abs_dsteer", "mean_abs_dthrottle",
            "mean_speed_ms", "std_speed_ms",
        ]

        groups = defaultdict(list)
        for r in all_summary_rows:
            k = tuple(r[g] for g in group_keys)
            groups[k].append(r)

        out_path = out_root / "eval_summary_mean_std.csv"
        with out_path.open("w", newline="") as f:
            fieldnames = group_keys + [f"{m}_mean" for m in metric_keys] + [f"{m}_std" for m in metric_keys]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for k, rows in groups.items():
                out = {group_keys[i]: k[i] for i in range(len(group_keys))}
                for mk in metric_keys:
                    vals = [float(rr.get(mk, 0.0)) for rr in rows]
                    out[f"{mk}_mean"] = statistics.fmean(vals) if vals else 0.0
                    out[f"{mk}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                w.writerow(out)

    finally:
        merged_f.close()
        summary_f.close()

if __name__ == "__main__":
    main()