#!/usr/bin/env python3
"""Aggregate experiment results, calibration outputs and produce summary CSV/JSON.

Usage: python tools/generate_experiment_report.py --experiments_dir ./experiments --out_dir ./experiments/report
"""
import argparse
from pathlib import Path
import json
import pandas as pd


def find_experiments(experiments_dir: Path):
    rows = []
    for d in experiments_dir.glob('finetune_*'):
        # expect a metrics.jsonl or metrics.csv in d
        metrics = d / 'metrics.jsonl'
        if not metrics.exists():
            # try common alternatives
            metrics = d / 'metrics.csv'
        rows.append({'dir': str(d), 'metrics': str(metrics)})
    return rows


def aggregate(experiments_dir: Path, out_dir: Path):
    rows = find_experiments(experiments_dir)
    out = []
    for r in rows:
        d = Path(r['dir'])
        metrics_file = Path(r['metrics'])
        if metrics_file.exists():
            try:
                df = pd.read_json(metrics_file, lines=True)
            except Exception:
                try:
                    df = pd.read_csv(metrics_file)
                except Exception:
                    continue
            # pick last row as final
            last = df.iloc[-1].to_dict()
            summary = {'dir': str(d)}
            # extract common fields if present
            for key in ['episode/score','episode/violation_rate','train/extr_reward_mean','train/expl_disag_reward_mean']:
                if key in last:
                    summary[key] = float(last[key])
            out.append(summary)
    out_df = pd.DataFrame(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / 'experiment_summary.csv', index=False)
    out_dir.joinpath('experiment_summary.json').write_text(out_df.to_json(orient='records', indent=2))
    print('Wrote summary to', out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiments_dir', required=True)
    p.add_argument('--out_dir', required=True)
    args = p.parse_args()
    aggregate(Path(args.experiments_dir), Path(args.out_dir))


if __name__ == '__main__':
    main()
