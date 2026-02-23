#!/usr/bin/env python3
"""
Generate a lightweight report (PNG plots + CSV summary) from the experiments folder.

It looks for `metrics.jsonl` files under each experiment logdir and per-episode CSVs
created by `tools/log_eval_episode.py` and summarizes/plots:
 - Pretrain: intrinsic disagreement reward curve
 - Finetune: episode returns per task/shield variant
 - Eval: override rates and mean speeds from CSVs

Usage:
  python tools/generate_report.py --expdir ./experiments --outdir ./reports

"""
import argparse
import json
import os
import glob
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt


def read_metrics_jsonl(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
    except Exception:
        return None
    return rows


def find_metrics(expdir):
    # Walk experiment dir for metrics.jsonl files
    metrics = {}
    for root, dirs, files in os.walk(expdir):
        for fn in files:
            if fn.endswith('metrics.jsonl'):
                path = os.path.join(root, fn)
                metrics[root] = read_metrics_jsonl(path)
    return metrics


def plot_pretrain_intrinsic(metrics, outdir):
    # Find pretrain folder (contains 'pretrain_ld') and plot disagreement intrinsic
    for root, rows in metrics.items():
        if 'pretrain_ld' in root and rows:
            # extract a plausible key name for intrinsic disag reward: 'train/extr' or 'train/expl_rewards.disag'
            steps = []
            disag = []
            for r in rows:
                steps.append(r.get('step', None) or r.get('train/step', None) or r.get('timer/step', None))
                # heuristic keys
                v = None
                for k in ('train/expl_rewards.disag', 'expl_rewards/extr', 'expl_rewards/disag'):
                    if k in r:
                        v = r[k]
                        break
                # fallback search
                if v is None:
                    for k in r:
                        if 'disag' in k.lower():
                            v = r[k]
                            break
                disag.append(v)
            df = pd.DataFrame({'step': steps, 'disag': disag}).dropna()
            if df.empty:
                continue
            plt.figure()
            plt.plot(df['step'], df['disag'])
            plt.xlabel('step')
            plt.ylabel('intrinsic_disag')
            plt.title('Pretrain intrinsic disagreement')
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(os.path.join(outdir, 'pretrain_intrinsic.png'))
            plt.close()
            print('Wrote', os.path.join(outdir, 'pretrain_intrinsic.png'))


def plot_finetune_returns(metrics, expdir, outdir):
    # For each finetune_* folder, read metrics and plot episode/score over steps
    rows = []
    for root, data in metrics.items():
        if '/finetune_' in root and data:
            scores = []
            steps = []
            for r in data:
                s = None
                for k in ('episode/score', 'episode/score', 'score'):
                    if k in r:
                        s = r[k]
                        break
                if s is None:
                    # try scanning
                    for k in r:
                        if 'score' in k:
                            s = r[k]
                            break
                scores.append(s)
                steps.append(r.get('step', None))
            df = pd.DataFrame({'step': steps, 'score': scores}).dropna()
            if df.empty:
                continue
            label = os.path.basename(root)
            rows.append((label, df))

    # plot all finetune curves
    if rows:
        plt.figure()
        for label, df in rows:
            plt.plot(df['step'], df['score'], label=label)
        plt.xlabel('step')
        plt.ylabel('episode return')
        plt.legend()
        plt.title('Finetune returns')
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, 'finetune_returns.png'))
        plt.close()
        print('Wrote', os.path.join(outdir, 'finetune_returns.png'))


def summarize_eval_csvs(expdir, outdir):
    # Parse eval CSVs using the parse_eval_results helper logic
    csvs = glob.glob(os.path.join(expdir, '**', '*.csv'), recursive=True)
    if not csvs:
        print('No eval CSVs found')
        return
    summary_rows = []
    for c in sorted(csvs):
        try:
            df = pd.read_csv(c)
        except Exception:
            continue
        n = len(df)
        shield_count = 0
        if 'shield_unsafe' in df.columns:
            shield_count = (df['shield_unsafe'].astype(float) > 0).sum()
        mean_speed = df['speed_kmh'].astype(float).mean() if 'speed_kmh' in df.columns else None
        total_reward = df['reward'].astype(float).sum() if 'reward' in df.columns else None
        summary_rows.append({'file': c, 'steps': n, 'override_rate': shield_count / n if n else 0.0, 'mean_speed_kmh': mean_speed, 'total_reward': total_reward})
    out_csv = os.path.join(outdir, 'eval_summary.csv')
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print('Wrote', out_csv)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--expdir', required=True)
    p.add_argument('--outdir', required=True)
    args = p.parse_args()

    metrics = find_metrics(args.expdir)
    plot_pretrain_intrinsic(metrics, args.outdir)
    plot_finetune_returns(metrics, args.expdir, args.outdir)
    summarize_eval_csvs(args.expdir, args.outdir)


if __name__ == '__main__':
    main()
