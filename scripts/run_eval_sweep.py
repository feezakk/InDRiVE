# #!/usr/bin/env python3
# import argparse
# import csv
# import json
# import os
# import subprocess
# from pathlib import Path
# from typing import Dict, List, Any

# def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
#     with path.open("r", newline="") as f:
#         return list(csv.DictReader(f))

# def _safe_float(x: Any, default: float = 0.0) -> float:
#     try:
#         return float(x)
#     except Exception:
#         return default

# def _summarize_episode_csv(rows: List[Dict[str, str]]) -> Dict[str, float]:
#     # expects columns: success, collision, off_road, out_of_lane, wrong_direction, too_slow, time_exceeded, destination_reached
#     if not rows:
#         return {"episodes": 0, "sr": 0.0}
#     n = len(rows)
#     sr = sum(int(_safe_float(r.get("success", 0))) for r in rows) / n

#     def mean(key: str) -> float:
#         return sum(_safe_float(r.get(key, 0.0)) for r in rows) / n

#     out = {
#         "episodes": float(n),
#         "sr": 100.0 * sr,
#         "collision_rate": 100.0 * mean("collision"),
#         "off_road_rate": 100.0 * mean("off_road"),
#         "out_of_lane_rate": 100.0 * mean("out_of_lane"),
#         "wrong_direction_rate": 100.0 * mean("wrong_direction"),
#         "too_slow_rate": 100.0 * mean("too_slow"),
#         "time_exceeded_rate": 100.0 * mean("time_exceeded"),

#         # comfort summaries if present
#         "mean_abs_acc_ms2": mean("mean_abs_acc_ms2"),
#         "mean_abs_jerk_ms3": mean("mean_abs_jerk_ms3"),
#         "mean_abs_dsteer": mean("mean_abs_dsteer"),
#         "mean_abs_dthrottle": mean("mean_abs_dthrottle"),
#         "mean_abs_lat_acc_ms2": mean("mean_abs_lat_acc_ms2"),
#         "mean_speed_ms": mean("mean_speed_ms"),
#         "std_speed_ms": mean("std_speed_ms"),
#     }
#     return out

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--carla_port", type=int, required=True)
#     ap.add_argument("--gpu", type=str, required=True)
#     ap.add_argument("--eval_sh", type=str, required=True, help="Path to your eval bash wrapper")
#     ap.add_argument("--routes_json", type=str, required=True, help="Route groups JSON (straight/turn/multi_turn)")
#     ap.add_argument("--spec_json", type=str, required=True, help="Sweep spec JSON (methods/phases/seeds/checkpoints)")
#     ap.add_argument("--out_root", type=str, required=True)
#     args, unknown = ap.parse_known_args()

#     out_root = Path(args.out_root)
#     out_root.mkdir(parents=True, exist_ok=True)

#     routes = json.loads(Path(args.routes_json).read_text())
#     spec = json.loads(Path(args.spec_json).read_text())
#     seeds = spec["seeds"]
#     phases = spec["phases"]
#     methods = spec["methods"]
#     eval_steps = int(spec.get("eval", {}).get("steps", 20000))
#     eval_episodes = int(spec.get("eval", {}).get("episodes", 20))
#     eval_densities = spec.get("eval", {}).get("densities", [5, 10, 20])
#     eval_tasks = spec.get("eval", {}).get("tasks", [])
#     if not eval_tasks:
#         raise ValueError("spec_json eval.tasks is required")

#     # merged outputs
#     merged_episode_csv = out_root / "eval_episodes_all.csv"
#     summary_csv = out_root / "eval_summary.csv"

#     merged_fields_written = merged_episode_csv.exists()
#     summary_fields_written = summary_csv.exists()

#     # We will append to merged CSV and summary CSV
#     merged_f = merged_episode_csv.open("a", newline="")
#     summary_f = summary_csv.open("a", newline="")

#     merged_writer = None
#     summary_writer = None

#     try:
#         import os

#         import statistics
#         from collections import defaultdict

#         all_summary_rows = []

#         for method_name, ckpt_map in methods.items():
#             for phase in phases:
#                 if phase not in ckpt_map:
#                     continue
#                 ckpt_tmpl = ckpt_map[phase]
#                 if not ckpt_tmpl:
#                     continue

#                 # stage used by RoutePoolMixin (must exist)
#                 stage = "zero_shot" if phase == "zero_shot" else "eval"

#                 for task in eval_tasks:
#                     if task not in routes:
#                         raise KeyError(f"routes_json missing task '{task}'")
#                     task_routes = routes[task]

#                     # for dens in eval_densities:
#                     #     for route_type, route_indices in task_routes.items():
#                     #         for route_idx in route_indices:
#                     #             for seed in seeds:
#                     #                 ckpt = ckpt_tmpl.format(seed=seed)
#                     #                 ckpt_path = Path(ckpt)
#                     #                 if not ckpt_path.exists():
#                     #                     raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

#                     #                 run_name = f"{task}__{method_name}__{phase}__dens{dens}__{route_type}__r{route_idx}__seed{seed}"
#                     #                 logdir = out_root / "runs" / run_name
#                     #                 logdir.mkdir(parents=True, exist_ok=True)

#                     #                 cmd = [
#                     #                     str(Path(args.eval_sh).resolve()),
#                     #                     str(args.carla_port),
#                     #                     str(args.gpu),
#                     #                     str(ckpt_path.resolve()),
#                     #                     run_name,
#                     #                     "--task", task,
#                     #                     "--dreamerv3.logdir", str(logdir),
                                        
#                     #                     "--dreamerv3.run.steps", str(eval_steps),
#                     #                     "--env.stage", stage,
#                     #                     "--env.lane_pair_index", str(route_idx),
#                     #                     "--env.num_vehicles", str(dens),
#                     #                 ] + unknown

#                     for route_type in task_routes.keys():   # now should be: straight, right, left, multi_turn
#                         for seed in seeds:
#                             ckpt = ckpt_tmpl.format(seed=seed)
#                             ckpt_path = Path(ckpt)
#                             if not ckpt_path.exists():
#                                 raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
#                             run_name = f"{task}__{method_name}__{phase}__{route_type}__seed{seed}"
#                             logdir = out_root / "runs" / run_name

#                             cmd = [
#                                 str(Path(args.eval_sh).resolve()),
#                                 str(args.carla_port),
#                                 str(args.gpu),
#                                 str(ckpt_path.resolve()),
#                                 run_name,
#                                 "--task", task,
#                                 "--dreamerv3.logdir", str(logdir),
#                                 "--dreamerv3.run.steps", str(eval_steps),
#                                 "--env.stage", stage,
#                                 "--env.route_group", str(route_type),   # <<<<< this is the new control knob
#                             ] + unknown

#                             print("[RUN]", " ".join(cmd), flush=True)

#                             env_vars = dict(os.environ)
#                             env_vars["EVAL_EPISODES"] = str(eval_episodes)
#                             subprocess.run(cmd, check=True, env=env_vars)



#                             # Consume per-episode CSV produced by eval script
#                             ep_csv = logdir / "eval_episode_metrics.csv"
#                             if not ep_csv.exists():
#                                 raise FileNotFoundError(
#                                     f"Expected {ep_csv} to exist. "
#                                     f"Your eval code must write eval_episode_metrics.csv."
#                                 )

#                             ep_rows = _read_csv_rows(ep_csv)

#                             # Write merged per-episode file with metadata columns
#                             for r in ep_rows:
#                                 r2 = dict(r)
#                                 r2.update({
#                                     "method": method_name,
#                                     "phase": phase,
#                                     "route_type": route_type,
#                                     # "route_idx": str(route_idx),
#                                     "seed": str(seed),
#                                     "checkpoint": str(ckpt_path),
#                                     "logdir": str(logdir),
#                                     "task": task,
#                                     # "traffic_density": str(dens),
#                                 })
#                                 if merged_writer is None:
#                                     merged_writer = csv.DictWriter(merged_f, fieldnames=list(r2.keys()))
#                                     if not merged_fields_written:
#                                         merged_writer.writeheader()
#                                         merged_fields_written = True
#                                 merged_writer.writerow(r2)
#                             merged_f.flush()

#                             # Write per-run summary row
#                             summ = _summarize_episode_csv(ep_rows)
#                             summ_row = {
#                                 "method": method_name,
#                                 "phase": phase,
#                                 "route_type": route_type,
#                                 # "route_idx": int(route_idx),
#                                 "seed": int(seed),
#                                 "checkpoint": str(ckpt_path),
#                                 "logdir": str(logdir),
#                                 "task": task,
#                                 # "traffic_density": int(dens),
#                                 **summ,
#                             }
#                             all_summary_rows.append(summ_row)
#                             if summary_writer is None:
#                                 summary_writer = csv.DictWriter(summary_f, fieldnames=list(summ_row.keys()))
#                                 if not summary_fields_written:
#                                     summary_writer.writeheader()
#                                     summary_fields_written = True
#                             summary_writer.writerow(summ_row)
#                             summary_f.flush()

#         print(f"[DONE] Wrote: {merged_episode_csv} and {summary_csv}")

#         group_keys = ["method", "phase", "task", "route_type", "route_idx", "traffic_density"]

#         metric_keys = [
#             "sr",
#             "collision_rate", "off_road_rate", "out_of_lane_rate",
#             "wrong_direction_rate", "too_slow_rate", "time_exceeded_rate",
#             "mean_abs_acc_ms2", "mean_abs_jerk_ms3", "mean_abs_lat_acc_ms2",
#             "mean_abs_dsteer", "mean_abs_dthrottle",
#             "mean_speed_ms", "std_speed_ms",
#         ]

#         groups = defaultdict(list)
#         for r in all_summary_rows:
#             k = tuple(r[g] for g in group_keys)
#             groups[k].append(r)

#         out_path = out_root / "eval_summary_mean_std.csv"
#         with out_path.open("w", newline="") as f:
#             fieldnames = group_keys + [f"{m}_mean" for m in metric_keys] + [f"{m}_std" for m in metric_keys]
#             w = csv.DictWriter(f, fieldnames=fieldnames)
#             w.writeheader()
#             for k, rows in groups.items():
#                 out = {group_keys[i]: k[i] for i in range(len(group_keys))}
#                 for mk in metric_keys:
#                     vals = [float(rr.get(mk, 0.0)) for rr in rows]
#                     out[f"{mk}_mean"] = statistics.fmean(vals) if vals else 0.0
#                     out[f"{mk}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
#                 w.writerow(out)

#     finally:
#         merged_f.close()
#         summary_f.close()

# if __name__ == "__main__":
#     main()




# # !/usr/bin/env python3
# import argparse
# import csv
# import json
# import os
# import shutil
# import subprocess
# from dataclasses import dataclass
# from datetime import datetime
# from pathlib import Path
# from typing import Dict, List, Any, Optional, Tuple
# import statistics
# from collections import defaultdict


# def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
#     with path.open("r", newline="") as f:
#         return list(csv.DictReader(f))


# def _safe_float(x: Any, default: float = 0.0) -> float:
#     try:
#         return float(x)
#     except Exception:
#         return default


# def _count_episode_rows(ep_csv: Path) -> int:
#     try:
#         with ep_csv.open("r", newline="") as f:
#             return sum(1 for _ in csv.DictReader(f))
#     except Exception:
#         return 0


# def _run_is_complete(logdir: Path, expect_episodes: int) -> Tuple[bool, str]:
#     ep_csv = logdir / "eval_episode_metrics.csv"
#     if not ep_csv.exists():
#         return False, "missing eval_episode_metrics.csv"
#     n = _count_episode_rows(ep_csv)
#     if n >= expect_episodes:
#         return True, f"complete ({n}/{expect_episodes})"
#     return False, f"incomplete ({n}/{expect_episodes})"


# def _summarize_episode_csv(rows: List[Dict[str, str]]) -> Dict[str, float]:
#     # expects columns: success, collision, off_road, out_of_lane, wrong_direction, too_slow, time_exceeded, destination_reached
#     if not rows:
#         return {"episodes": 0.0, "sr": 0.0}

#     n = len(rows)
#     sr = sum(int(_safe_float(r.get("success", 0))) for r in rows) / n

#     def mean(key: str) -> float:
#         return sum(_safe_float(r.get(key, 0.0)) for r in rows) / n

#     out = {
#         "episodes": float(n),
#         "sr": 100.0 * sr,
#         "collision_rate": 100.0 * mean("collision"),
#         "off_road_rate": 100.0 * mean("off_road"),
#         "out_of_lane_rate": 100.0 * mean("out_of_lane"),
#         "wrong_direction_rate": 100.0 * mean("wrong_direction"),
#         "too_slow_rate": 100.0 * mean("too_slow"),
#         "time_exceeded_rate": 100.0 * mean("time_exceeded"),

#         # comfort summaries if present
#         "mean_abs_acc_ms2": mean("mean_abs_acc_ms2"),
#         "mean_abs_jerk_ms3": mean("mean_abs_jerk_ms3"),
#         "mean_abs_dsteer": mean("mean_abs_dsteer"),
#         "mean_abs_dthrottle": mean("mean_abs_dthrottle"),
#         "mean_abs_lat_acc_ms2": mean("mean_abs_lat_acc_ms2"),
#         "mean_speed_ms": mean("mean_speed_ms"),
#         "std_speed_ms": mean("std_speed_ms"),
#     }
#     return out


# @dataclass(frozen=True)
# class RunSpec:
#     method: str
#     phase: str
#     task: str
#     route_group: str
#     seed: int
#     checkpoint: str
#     stage: str
#     run_name: str
#     logdir: Path


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--carla_port", type=int, required=True)
#     ap.add_argument("--gpu", type=str, required=True)
#     ap.add_argument("--eval_sh", type=str, required=True, help="Path to your eval bash wrapper")
#     ap.add_argument("--routes_json", type=str, required=True, help="Route groups JSON (keys are route_group names)")
#     ap.add_argument("--spec_json", type=str, required=True, help="Sweep spec JSON (methods/phases/seeds/checkpoints)")
#     ap.add_argument("--out_root", type=str, required=True)

#     # resume controls
#     ap.add_argument("--resume", action="store_true", default=True,
#                     help="Skip runs that already have a complete eval_episode_metrics.csv")
#     ap.add_argument("--no-resume", dest="resume", action="store_false",
#                     help="Force re-run everything (danger: expensive)")
#     ap.add_argument("--clean_incomplete", action="store_true", default=True,
#                     help="If a run has an incomplete CSV, rename the folder and rerun cleanly.")
#     ap.add_argument("--keep_incomplete_suffix", type=str, default="_INCOMPLETE_",
#                     help="Suffix used when renaming incomplete run folders.")

#     args, unknown = ap.parse_known_args()

#     out_root = Path(args.out_root)
#     runs_root = out_root / "runs"
#     runs_root.mkdir(parents=True, exist_ok=True)

#     routes = json.loads(Path(args.routes_json).read_text())
#     spec = json.loads(Path(args.spec_json).read_text())

#     seeds: List[int] = list(spec["seeds"])
#     phases: List[str] = list(spec["phases"])
#     methods: Dict[str, Dict[str, str]] = dict(spec["methods"])

#     eval_steps = int(spec.get("eval", {}).get("steps", 20000))
#     eval_episodes = int(spec.get("eval", {}).get("episodes", 20))
#     eval_tasks = list(spec.get("eval", {}).get("tasks", []))
#     if not eval_tasks:
#         raise ValueError("spec_json eval.tasks is required")

#     # -------- Build full run list (deterministic ordering) --------
#     all_runs: List[RunSpec] = []
#     for method_name, ckpt_map in methods.items():
#         for phase in phases:
#             if phase not in ckpt_map:
#                 continue
#             ckpt_tmpl = ckpt_map[phase]
#             if not ckpt_tmpl:
#                 continue

#             stage = "zero_shot" if phase == "zero_shot" else "eval"

#             for task in eval_tasks:
#                 if task not in routes:
#                     raise KeyError(f"routes_json missing task '{task}'")
#                 task_routes = routes[task]

#                 # route_group keys are whatever you put in routes.json (straight/right/left/multi_turn)
#                 for route_group in sorted(task_routes.keys()):
#                     for seed in seeds:
#                         ckpt = ckpt_tmpl.format(seed=seed)
#                         ckpt_path = Path(ckpt)
#                         if not ckpt_path.exists():
#                             raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

#                         run_name = f"{task}__{method_name}__{phase}__{route_group}__seed{seed}"
#                         logdir = runs_root / run_name
#                         all_runs.append(RunSpec(
#                             method=method_name,
#                             phase=phase,
#                             task=task,
#                             route_group=route_group,
#                             seed=int(seed),
#                             checkpoint=str(ckpt_path.resolve()),
#                             stage=stage,
#                             run_name=run_name,
#                             logdir=logdir,
#                         ))

#     # -------- Execute with resume --------
#     failed: List[Tuple[RunSpec, str]] = []
#     completed: List[RunSpec] = []

#     for rs in all_runs:
#         done, msg = _run_is_complete(rs.logdir, eval_episodes)

#         if args.resume and done:
#             print(f"[SKIP] {rs.run_name} :: {msg}", flush=True)
#             completed.append(rs)
#             continue

#         if rs.logdir.exists() and args.clean_incomplete:
#             ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#             new_name = rs.logdir.with_name(rs.logdir.name + f"{args.keep_incomplete_suffix}{ts}")
#             print(f"[CLEAN] Renaming incomplete run dir:\n"
#                   f"        {rs.logdir}\n"
#                   f"     -> {new_name}", flush=True)
#             try:
#                 rs.logdir.rename(new_name)
#             except Exception:
#                 # fallback: delete if rename fails
#                 shutil.rmtree(rs.logdir, ignore_errors=True)

#         rs.logdir.mkdir(parents=True, exist_ok=True)

#         cmd = [
#             str(Path(args.eval_sh).resolve()),
#             str(args.carla_port),
#             str(args.gpu),
#             rs.checkpoint,
#             rs.run_name,
#             "--task", rs.task,
#             "--dreamerv3.logdir", str(rs.logdir),
#             "--dreamerv3.run.steps", str(eval_steps),
#             "--env.stage", rs.stage,
#             "--env.route_group", rs.route_group,
#         ] + unknown

#         print("[RUN]", " ".join(cmd), flush=True)

#         env_vars = dict(os.environ)
#         env_vars["EVAL_EPISODES"] = str(eval_episodes)

#         # Your wrapper currently exits 0 even on CARLA crash.
#         # So we cannot rely on returncode; we rely on existence/completeness of CSV.
#         try:
#             subprocess.run(cmd, check=False, env=env_vars)
#         except Exception as e:
#             failed.append((rs, f"subprocess exception: {e}"))
#             continue

#         done2, msg2 = _run_is_complete(rs.logdir, eval_episodes)
#         if not done2:
#             failed.append((rs, msg2))
#             print(f"[FAIL] {rs.run_name} :: {msg2}", flush=True)
#         else:
#             completed.append(rs)
#             print(f"[OK]   {rs.run_name} :: {msg2}", flush=True)

#     # -------- Rebuild merged outputs from completed runs (no duplicates) --------
#     merged_episode_csv = out_root / "eval_episodes_all.csv"
#     summary_csv = out_root / "eval_summary.csv"
#     meanstd_csv = out_root / "eval_summary_mean_std.csv"

#     all_summary_rows: List[Dict[str, Any]] = []

#     with merged_episode_csv.open("w", newline="") as mf, summary_csv.open("w", newline="") as sf:
#         merged_writer = None
#         summary_writer = None

#         for rs in completed:
#             ep_csv = rs.logdir / "eval_episode_metrics.csv"
#             if not ep_csv.exists():
#                 continue
#             ep_rows = _read_csv_rows(ep_csv)

#             # merged per-episode
#             for r in ep_rows:
#                 r2 = dict(r)
#                 r2.update({
#                     "method": rs.method,
#                     "phase": rs.phase,
#                     "task": rs.task,
#                     "route_group": rs.route_group,
#                     "seed": str(rs.seed),
#                     "checkpoint": rs.checkpoint,
#                     "logdir": str(rs.logdir),
#                     "run_name": rs.run_name,
#                 })
#                 if merged_writer is None:
#                     merged_writer = csv.DictWriter(mf, fieldnames=list(r2.keys()))
#                     merged_writer.writeheader()
#                 merged_writer.writerow(r2)

#             # per-run summary
#             summ = _summarize_episode_csv(ep_rows)
#             summ_row = {
#                 "method": rs.method,
#                 "phase": rs.phase,
#                 "task": rs.task,
#                 "route_group": rs.route_group,
#                 "seed": int(rs.seed),
#                 "checkpoint": rs.checkpoint,
#                 "logdir": str(rs.logdir),
#                 "run_name": rs.run_name,
#                 **summ,
#             }
#             all_summary_rows.append(summ_row)

#             if summary_writer is None:
#                 summary_writer = csv.DictWriter(sf, fieldnames=list(summ_row.keys()))
#                 summary_writer.writeheader()
#             summary_writer.writerow(summ_row)

#     # mean/std across seeds (grouping)
#     group_keys = ["method", "phase", "task", "route_group"]
#     metric_keys = [
#         "sr",
#         "collision_rate", "off_road_rate", "out_of_lane_rate",
#         "wrong_direction_rate", "too_slow_rate", "time_exceeded_rate",
#         "mean_abs_acc_ms2", "mean_abs_jerk_ms3", "mean_abs_lat_acc_ms2",
#         "mean_abs_dsteer", "mean_abs_dthrottle",
#         "mean_speed_ms", "std_speed_ms",
#         "episodes",
#     ]

#     groups = defaultdict(list)
#     for r in all_summary_rows:
#         k = tuple(r[g] for g in group_keys)
#         groups[k].append(r)

#     with meanstd_csv.open("w", newline="") as f:
#         fieldnames = group_keys + [f"{m}_mean" for m in metric_keys] + [f"{m}_std" for m in metric_keys]
#         w = csv.DictWriter(f, fieldnames=fieldnames)
#         w.writeheader()

#         for k, rows in groups.items():
#             out = {group_keys[i]: k[i] for i in range(len(group_keys))}
#             for mk in metric_keys:
#                 vals = [float(rr.get(mk, 0.0)) for rr in rows]
#                 out[f"{mk}_mean"] = statistics.fmean(vals) if vals else 0.0
#                 out[f"{mk}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
#             w.writerow(out)

#     print(f"\n[DONE] wrote:\n  {merged_episode_csv}\n  {summary_csv}\n  {meanstd_csv}", flush=True)

#     if failed:
#         print("\n[FAILED RUNS] (you can re-run and it will resume):", flush=True)
#         for rs, reason in failed:
#             print(f"  - {rs.run_name} :: {reason}", flush=True)


# if __name__ == "__main__":
#     main()




#!/usr/bin/env python3
# import argparse
# import csv
# import json
# import os
# import shutil
# import subprocess
# from pathlib import Path
# from typing import Dict, List, Any, Tuple

# def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
#     with path.open("r", newline="") as f:
#         return list(csv.DictReader(f))

# def _safe_float(x: Any, default: float = 0.0) -> float:
#     try:
#         return float(x)
#     except Exception:
#         return default

# def _progress_from_episode_csv(ep_csv: Path) -> Tuple[int, int]:
#     """
#     Returns (episodes_done, max_env_step).
#     env_step is written by eval_safety.py into eval_episode_metrics.csv.
#     """
#     if not ep_csv.exists():
#         return 0, 0
#     rows = _read_csv_rows(ep_csv)
#     if not rows:
#         return 0, 0
#     episodes_done = len(rows)
#     max_env_step = 0
#     for r in rows:
#         max_env_step = max(max_env_step, int(_safe_float(r.get("env_step", 0), 0.0)))
#     return episodes_done, max_env_step

# def _is_complete(logdir: Path, eval_episodes: int, eval_steps: int) -> bool:
#     ep_csv = logdir / "eval_episode_metrics.csv"
#     episodes_done, max_env_step = _progress_from_episode_csv(ep_csv)
#     # Matches eval_safety loop: stop when episodes OR steps limit reached
#     return (episodes_done >= eval_episodes) or (max_env_step >= eval_steps)

# def _summarize_episode_csv(rows: List[Dict[str, str]]) -> Dict[str, float]:
#     if not rows:
#         return {"episodes": 0.0, "sr": 0.0}
#     n = len(rows)

#     def mean(key: str) -> float:
#         return sum(_safe_float(r.get(key, 0.0)) for r in rows) / n

#     sr = mean("success") * 100.0

#     out = {
#         "episodes": float(n),
#         "sr": sr,
#         "collision_rate": mean("collision") * 100.0,
#         "off_road_rate": mean("off_road") * 100.0,
#         "out_of_lane_rate": mean("out_of_lane") * 100.0,
#         "wrong_direction_rate": mean("wrong_direction") * 100.0,
#         "too_slow_rate": mean("too_slow") * 100.0,
#         "time_exceeded_rate": mean("time_exceeded") * 100.0,
#         "destination_reached_rate": mean("destination_reached") * 100.0,

#         "mean_abs_acc_ms2": mean("mean_abs_acc_ms2"),
#         "mean_abs_jerk_ms3": mean("mean_abs_jerk_ms3"),
#         "mean_abs_dsteer": mean("mean_abs_dsteer"),
#         "mean_abs_dthrottle": mean("mean_abs_dthrottle"),
#         "mean_abs_lat_acc_ms2": mean("mean_abs_lat_acc_ms2"),
#         "mean_speed_ms": mean("mean_speed_ms"),
#         "std_speed_ms": mean("std_speed_ms"),
#     }
#     return out

# def _is_complete(logdir: Path, target_steps: int) -> bool:
#     done = logdir / "eval_done.json"
#     if not done.exists():
#         return False
#     try:
#         import json
#         d = json.loads(done.read_text())
#         return bool(d.get("ok", False)) and int(d.get("final_step", 0)) >= int(target_steps)
#     except Exception:
#         return False

# def _clean_incomplete(logdir: Path) -> None:
#     # Avoid appending to old CSVs
#     for fn in ["eval_episode_metrics.csv", "eval_done.json", "metrics.jsonl"]:
#         p = logdir / fn
#         if p.exists():
#             p.unlink()

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--carla_port", type=int, required=True)
#     ap.add_argument("--gpu", type=str, required=True)
#     ap.add_argument("--eval_sh", type=str, required=True)
#     ap.add_argument("--routes_json", type=str, required=True)
#     ap.add_argument("--spec_json", type=str, required=True)
#     ap.add_argument("--out_root", type=str, required=True)

#     ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
#     ap.add_argument("--max_retries", type=int, default=2)
#     ap.add_argument("--continue_on_fail", action=argparse.BooleanOptionalAction, default=True)
#     ap.add_argument("--keep_failed_logdirs", action=argparse.BooleanOptionalAction, default=True)

#     args, unknown = ap.parse_known_args()

#     out_root = Path(args.out_root)
#     runs_root = out_root / "runs"
#     runs_root.mkdir(parents=True, exist_ok=True)

#     routes = json.loads(Path(args.routes_json).read_text())
#     spec = json.loads(Path(args.spec_json).read_text())

#     seeds = spec["seeds"]
#     phases = spec["phases"]
#     methods = spec["methods"]

#     eval_steps = int(spec.get("eval", {}).get("steps", 20000))
#     eval_episodes = int(spec.get("eval", {}).get("episodes", 20))
#     eval_tasks = spec.get("eval", {}).get("tasks", [])
#     if not eval_tasks:
#         raise ValueError("spec_json eval.tasks is required")

#     failures = []

#     def _maybe_archive_logdir(logdir: Path):
#         if not logdir.exists():
#             return
#         if not args.keep_failed_logdirs:
#             shutil.rmtree(logdir, ignore_errors=True)
#             return
#         # archive it
#         i = 0
#         while True:
#             cand = logdir.parent / f"{logdir.name}__FAILED{i}"
#             if not cand.exists():
#                 shutil.move(str(logdir), str(cand))
#                 return
#             i += 1

#     def _is_complete(logdir: Path, target_steps: int) -> bool:
#         done = logdir / "eval_done.json"
#         if not done.exists():
#             return False
#         try:
#             import json
#             d = json.loads(done.read_text())
#             return bool(d.get("ok", False)) and int(d.get("final_step", 0)) >= int(target_steps)
#         except Exception:
#             return False

#     def _clean_incomplete(logdir: Path) -> None:
#         # Avoid appending to old CSVs
#         for fn in ["eval_episode_metrics.csv", "eval_done.json", "metrics.jsonl"]:
#             p = logdir / fn
#             if p.exists():
#                 p.unlink()

#     merged_writer = None

#     # -------------------------
#     # Run all evaluations (resume-aware)
#     # -------------------------
#     for method_name, ckpt_map in methods.items():
#         for phase in phases:
#             if phase not in ckpt_map:
#                 continue
#             ckpt_tmpl = ckpt_map[phase]
#             if not ckpt_tmpl:
#                 continue

#             stage = "zero_shot" if phase == "zero_shot" else "eval"

#             for task in eval_tasks:
#                 task_routes = routes[task]
#                 route_groups = list(task_routes.keys())  # straight/right/left/multi_turn

#                 for route_group in route_groups:
#                     for seed in seeds:
#                         ckpt = ckpt_tmpl.format(seed=seed)
#                         ckpt_path = Path(ckpt)
#                         if not ckpt_path.exists():
#                             print(f"[SKIP] missing checkpoint: {ckpt_path}")
#                             continue

#                         run_name = f"{task}__{method_name}__{phase}__{route_group}__seed{seed}"
#                         logdir = out_root / "runs" / run_name
#                         logdir.mkdir(parents=True, exist_ok=True)

#                         # RESUME: skip completed
#                         if _is_complete(logdir, eval_steps):
#                             print(f"[SKIP] complete: {run_name}")
#                             continue

#                         # If partial outputs exist, clean them so we don't append mixed runs
#                         _clean_incomplete(logdir)

#                         cmd = [
#                             str(Path(args.eval_sh).resolve()),
#                             str(args.carla_port),
#                             str(args.gpu),
#                             str(ckpt_path.resolve()),
#                             run_name,
#                             "--task", task,
#                             "--dreamerv3.logdir", str(logdir),
#                             "--dreamerv3.run.steps", str(eval_steps),
#                             "--env.stage", stage,
#                             "--env.route_group", str(route_group),
#                         ]

#                         print("[RUN]", " ".join(cmd), flush=True)

#                         env_vars = dict(os.environ)
#                         # You can still set this for logging/reference, but eval will not stop on it anymore
#                         env_vars["EVAL_EPISODES"] = str(eval_episodes)

#                         # IMPORTANT: do NOT use check=True; CARLA can segfault
#                         proc = subprocess.run(cmd, env=env_vars)
#                         rc = proc.returncode

#                         # Validate completion (done file + steps)
#                         if rc == 0 and _is_complete(logdir, eval_steps):
#                             print(f"[OK] {run_name}")
#                         else:
#                             print(f"[FAIL] {run_name} rc={rc} (missing/invalid eval_done.json)")
#                             # optionally write to failures.csv and continue
#                             continue

#                         # Now safely read eval_episode_metrics.csv and write merged/summary
#                         ep_csv = logdir / "eval_episode_metrics.csv"
#                         if not ep_csv.exists():
#                             print(f"[FAIL] Missing {ep_csv} even though done.json exists?")
#                             continue

#                         ep_rows = _read_csv_rows(ep_csv)

#                         # Merge per-episode rows (NOTE: no fixed route_idx/dens anymore)
#                         for r in ep_rows:
#                             r2 = dict(r)
#                             r2.update({
#                                 "method": method_name,
#                                 "phase": phase,
#                                 "route_group": route_group,
#                                 "seed": str(seed),
#                                 "checkpoint": str(ckpt_path),
#                                 "logdir": str(logdir),
#                                 "task": task,
#                             })
#                             if merged_writer is None:
#                                 merged_writer = csv.DictWriter(merged_f, fieldnames=list(r2.keys()))
#                                 if not merged_fields_written:
#                                     merged_writer.writeheader()
#                                     merged_fields_written = True
#                             merged_writer.writerow(r2)
#                         merged_f.flush()

#                         summ = _summarize_episode_csv(ep_rows)
#                         summ_row = {
#                             "method": method_name,
#                             "phase": phase,
#                             "route_group": route_group,
#                             "seed": int(seed),
#                             "checkpoint": str(ckpt_path),
#                             "logdir": str(logdir),
#                             "task": task,
#                             **summ,
#                         }
#                         all_summary_rows.append(summ_row)
#                         if summary_writer is None:
#                             summary_writer = csv.DictWriter(summary_f, fieldnames=list(summ_row.keys()))
#                             if not summary_fields_written:
#                                 summary_writer.writeheader()
#                                 summary_fields_written = True
#                         summary_writer.writerow(summ_row)
#                         summary_f.flush()

#     # -------------------------
#     # Build merged outputs deterministically (no duplicates across resumes)
#     # -------------------------
#     merged_episode_csv = out_root / "eval_episodes_all.csv"
#     summary_csv = out_root / "eval_summary.csv"

#     with merged_episode_csv.open("w", newline="") as mf, summary_csv.open("w", newline="") as sf:
#         merged_writer = None
#         summary_writer = None

#         for logdir in sorted(runs_root.glob("*")):
#             ep_csv = logdir / "eval_episode_metrics.csv"
#             if not ep_csv.exists():
#                 continue
#             rows = _read_csv_rows(ep_csv)
#             if not rows:
#                 continue

#             # parse metadata from run_name (since you encode it)
#             # format: task__method__phase__route_group__seedX
#             parts = logdir.name.split("__")
#             if len(parts) < 5:
#                 continue
#             task = parts[0]
#             method = parts[1]
#             phase = parts[2]
#             route_group = parts[3]
#             seed_str = parts[4].replace("seed", "")
#             seed = int(_safe_float(seed_str, 0.0))

#             # merged per-episode rows
#             for r in rows:
#                 r2 = dict(r)
#                 r2.update({
#                     "task": task,
#                     "method": method,
#                     "phase": phase,
#                     "route_group": route_group,
#                     "seed": seed,
#                     "logdir": str(logdir),
#                 })
#                 if merged_writer is None:
#                     merged_writer = csv.DictWriter(mf, fieldnames=list(r2.keys()))
#                     merged_writer.writeheader()
#                 merged_writer.writerow(r2)

#             # per-run summary row
#             summ = _summarize_episode_csv(rows)
#             summ_row = {
#                 "task": task,
#                 "method": method,
#                 "phase": phase,
#                 "route_group": route_group,
#                 "seed": seed,
#                 "logdir": str(logdir),
#                 **summ,
#             }
#             if summary_writer is None:
#                 summary_writer = csv.DictWriter(sf, fieldnames=list(summ_row.keys()))
#                 summary_writer.writeheader()
#             summary_writer.writerow(summ_row)

#     # failures report
#     if failures:
#         fail_path = out_root / "failures.json"
#         fail_path.write_text(json.dumps(failures, indent=2))
#         print(f"[WARN] Some runs failed. See: {fail_path}", flush=True)

#     print(f"[DONE] Wrote: {merged_episode_csv} and {summary_csv}", flush=True)

# if __name__ == "__main__":
#     main()




#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import statistics
from collections import defaultdict


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _read_done_status(logdir: Path) -> Optional[Dict[str, Any]]:
    p = logdir / "eval_done.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _run_is_complete(logdir: Path, expect_steps: int) -> Tuple[bool, str]:
    st = _read_done_status(logdir)
    if st is None:
        return False, "missing eval_done.json"
    steps = int(st.get("steps", -1))
    if steps >= expect_steps:
        return True, f"complete (steps={steps} >= {expect_steps})"
    return False, f"incomplete (steps={steps} < {expect_steps})"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _summarize_episode_csv(rows: List[Dict[str, str]]) -> Dict[str, float]:
    if not rows:
        return {"episodes": 0.0, "sr": 0.0}

    n = len(rows)
    sr = sum(int(_safe_float(r.get("success", 0))) for r in rows) / n

    def mean(key: str) -> float:
        return sum(_safe_float(r.get(key, 0.0)) for r in rows) / n

    return {
        "episodes": float(n),
        "sr": 100.0 * sr,
        "collision_rate": 100.0 * mean("collision"),
        "off_road_rate": 100.0 * mean("off_road"),
        "out_of_lane_rate": 100.0 * mean("out_of_lane"),
        "wrong_direction_rate": 100.0 * mean("wrong_direction"),
        "too_slow_rate": 100.0 * mean("too_slow"),
        "time_exceeded_rate": 100.0 * mean("time_exceeded"),
        "mean_abs_acc_ms2": mean("mean_abs_acc_ms2"),
        "mean_abs_jerk_ms3": mean("mean_abs_jerk_ms3"),
        "mean_abs_dsteer": mean("mean_abs_dsteer"),
        "mean_abs_dthrottle": mean("mean_abs_dthrottle"),
        "mean_abs_lat_acc_ms2": mean("mean_abs_lat_acc_ms2"),
        "mean_speed_ms": mean("mean_speed_ms"),
        "std_speed_ms": mean("std_speed_ms"),
    }


@dataclass(frozen=True)
class RunSpec:
    method: str
    phase: str
    task: str
    route_group: str
    seed: int
    checkpoint: str
    stage: str
    run_name: str
    logdir: Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carla_port", type=int, required=True)
    ap.add_argument("--gpu", type=str, required=True)
    ap.add_argument("--eval_sh", type=str, required=True)
    ap.add_argument("--routes_json", type=str, required=True)
    ap.add_argument("--spec_json", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)

    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--clean_incomplete", action="store_true", default=True)
    ap.add_argument("--incomplete_suffix", type=str, default="_INCOMPLETE_")

    args = ap.parse_args()

    out_root = Path(args.out_root)
    runs_root = out_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    routes = json.loads(Path(args.routes_json).read_text())
    spec = json.loads(Path(args.spec_json).read_text())

    seeds: List[int] = list(spec["seeds"])
    phases: List[str] = list(spec["phases"])
    methods: Dict[str, Dict[str, str]] = dict(spec["methods"])

    eval_steps = int(spec.get("eval", {}).get("steps", 50000))
    eval_tasks = list(spec.get("eval", {}).get("tasks", []))
    if not eval_tasks:
        raise ValueError("spec_json eval.tasks is required")

    # ---- build all run specs ----
    all_runs: List[RunSpec] = []
    for method_name, ckpt_map in methods.items():
        for phase in phases:
            if phase not in ckpt_map:
                continue
            ckpt_tmpl = ckpt_map[phase]
            if not ckpt_tmpl:
                continue

            stage = "zero_shot" if phase == "zero_shot" else "eval"

            for task in eval_tasks:
                if task not in routes:
                    raise KeyError(f"routes_json missing task '{task}'")
                task_routes = routes[task]

                for route_group in sorted(task_routes.keys()):
                    for seed in seeds:
                        ckpt = ckpt_tmpl.format(seed=seed)
                        ckpt_path = Path(ckpt)
                        if not ckpt_path.exists():
                            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

                        run_name = f"{task}__{method_name}__{phase}__{route_group}__seed{seed}"
                        logdir = runs_root / run_name

                        all_runs.append(RunSpec(
                            method=method_name,
                            phase=phase,
                            task=task,
                            route_group=route_group,
                            seed=int(seed),
                            checkpoint=str(ckpt_path.resolve()),
                            stage=stage,
                            run_name=run_name,
                            logdir=logdir,
                        ))

    completed: List[RunSpec] = []
    failed: List[Tuple[RunSpec, str]] = []

    # ---- run with resume ----
    for rs in all_runs:
        done, msg = _run_is_complete(rs.logdir, eval_steps)

        if args.resume and done:
            print(f"[SKIP] {rs.run_name} :: {msg}", flush=True)
            completed.append(rs)
            continue

        if rs.logdir.exists() and args.clean_incomplete:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_dir = rs.logdir.with_name(rs.logdir.name + f"{args.incomplete_suffix}{ts}")
            print(f"[CLEAN] {rs.run_name}\n  {rs.logdir}\n  -> {new_dir}", flush=True)
            try:
                rs.logdir.rename(new_dir)
            except Exception:
                shutil.rmtree(rs.logdir, ignore_errors=True)

        rs.logdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(Path(args.eval_sh).resolve()),
            str(args.carla_port),
            str(args.gpu),
            rs.checkpoint,
            rs.run_name,
            "--task", rs.task,
            "--dreamerv3.logdir", str(rs.logdir),
            "--dreamerv3.run.steps", str(eval_steps),
            "--env.stage", rs.stage,
            "--env.route_group", rs.route_group,
        ]

        print("[RUN]", " ".join(cmd), flush=True)

        # IMPORTANT: do not set EVAL_EPISODES here if you changed eval_safety to be step-based.
        # If you keep episode-based stopping, you will again stop before 50k steps.
        env_vars = dict(os.environ)
        subprocess.run(cmd, check=False, env=env_vars)

        done2, msg2 = _run_is_complete(rs.logdir, eval_steps)
        if done2:
            print(f"[OK]   {rs.run_name} :: {msg2}", flush=True)
            completed.append(rs)
        else:
            print(f"[FAIL] {rs.run_name} :: {msg2}", flush=True)
            failed.append((rs, msg2))

    # ---- rebuild merged outputs from completed runs ----
    merged_episode_csv = out_root / "eval_episodes_all.csv"
    summary_csv = out_root / "eval_summary.csv"
    meanstd_csv = out_root / "eval_summary_mean_std.csv"

    all_summary_rows: List[Dict[str, Any]] = []

    with merged_episode_csv.open("w", newline="") as mf, summary_csv.open("w", newline="") as sf:
        merged_writer = None
        summary_writer = None

        for rs in completed:
            ep_csv = rs.logdir / "eval_episode_metrics.csv"
            if not ep_csv.exists():
                continue
            ep_rows = _read_csv_rows(ep_csv)

            for r in ep_rows:
                r2 = dict(r)
                r2.update({
                    "method": rs.method,
                    "phase": rs.phase,
                    "task": rs.task,
                    "route_group": rs.route_group,
                    "seed": str(rs.seed),
                    "checkpoint": rs.checkpoint,
                    "logdir": str(rs.logdir),
                    "run_name": rs.run_name,
                })
                if merged_writer is None:
                    merged_writer = csv.DictWriter(mf, fieldnames=list(r2.keys()))
                    merged_writer.writeheader()
                merged_writer.writerow(r2)

            summ = _summarize_episode_csv(ep_rows)
            summ_row = {
                "method": rs.method,
                "phase": rs.phase,
                "task": rs.task,
                "route_group": rs.route_group,
                "seed": int(rs.seed),
                **summ,
            }
            all_summary_rows.append(summ_row)

            if summary_writer is None:
                summary_writer = csv.DictWriter(sf, fieldnames=list(summ_row.keys()))
                summary_writer.writeheader()
            summary_writer.writerow(summ_row)

    # mean/std across seeds
    group_keys = ["method", "phase", "task", "route_group"]
    metric_keys = [
        "sr",
        "collision_rate", "off_road_rate", "out_of_lane_rate",
        "wrong_direction_rate", "too_slow_rate", "time_exceeded_rate",
        "mean_abs_acc_ms2", "mean_abs_jerk_ms3", "mean_abs_lat_acc_ms2",
        "mean_abs_dsteer", "mean_abs_dthrottle",
        "mean_speed_ms", "std_speed_ms",
        "episodes",
    ]

    groups = defaultdict(list)
    for r in all_summary_rows:
        groups[tuple(r[g] for g in group_keys)].append(r)

    with meanstd_csv.open("w", newline="") as f:
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

    print(f"\n[DONE] wrote:\n  {merged_episode_csv}\n  {summary_csv}\n  {meanstd_csv}", flush=True)

    if failed:
        print("\n[FAILED RUNS] (re-run and it will resume):", flush=True)
        for rs, reason in failed:
            print(f"  - {rs.run_name} :: {reason}", flush=True)


if __name__ == "__main__":
    main()