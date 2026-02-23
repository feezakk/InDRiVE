#!/usr/bin/env python3
"""Calibrate collision head predictions using temperature scaling and compute metrics.

Inputs: directory with per-episode CSVs produced by log_eval_episode.py (must contain columns
       'collision_pred' (logits or scores), and 'collision_label' (0/1) or similar).
Outputs: JSON with temperature, ECE, Brier, AUROC and thresholds->rates CSV.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def load_data(in_dir: Path):
    rows = []
    for p in sorted(in_dir.rglob('*.csv')):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if 'collision_pred' in df.columns and 'collision_label' in df.columns:
            rows.append(df[['collision_pred','collision_label']])
    if not rows:
        raise SystemExit('No CSVs with collision_pred/collision_label found in '+str(in_dir))
    data = pd.concat(rows, ignore_index=True)
    return data


def nll_temp_scale(logits, labels, temp):
    # logits -> prob via sigmoid(logits/temp)
    probs = 1.0 / (1.0 + np.exp(-logits / temp))
    # negative log-likelihood
    eps = 1e-12
    nll = -np.mean(labels * np.log(probs + eps) + (1-labels) * np.log(1-probs + eps))
    return nll


def fit_temperature(logits, labels):
    # simple grid search for temperature in [0.1,5]
    temps = np.concatenate([np.linspace(0.5,2.0,31), np.linspace(2.5,10.0,15)])
    best_t, best_nll = 1.0, float('inf')
    for t in temps:
        nll = nll_temp_scale(logits, labels, t)
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


def expected_calibration_error(probs, labels, n_bins=15):
    bins = np.linspace(0.0,1.0,n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        if mask.sum()==0:
            continue
        p_hat = probs[mask].mean()
        y_hat = labels[mask].mean()
        ece += (mask.sum()/ len(probs)) * abs(p_hat - y_hat)
    return float(ece)


def threshold_rates(probs, labels, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.0,1.0,21)
    rows = []
    for t in thresholds:
        pred_pos = probs >= t
        pred_rate = pred_pos.mean()
        true_pos = labels[pred_pos].mean() if pred_pos.sum()>0 else 0.0
        rows.append({'threshold': float(t), 'pred_rate': float(pred_rate), 'true_pos_rate': float(true_pos)})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--k', type=int, default=5)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_data(in_dir)
    logits = data['collision_pred'].to_numpy(dtype=float)
    labels = data['collision_label'].to_numpy(dtype=int)

    # split val/test deterministically by hash
    n = len(logits)
    idx = np.arange(n)
    # simple deterministic shuffle
    rng = np.random.RandomState(0)
    rng.shuffle(idx)
    n_val = max(1, int(0.1 * n))
    val_idx = idx[:n_val]
    test_idx = idx[n_val:]

    t_best = fit_temperature(logits[val_idx], labels[val_idx])
    probs_test = 1.0 / (1.0 + np.exp(-logits[test_idx] / t_best))

    ece = expected_calibration_error(probs_test, labels[test_idx])
    brier = float(brier_score_loss(labels[test_idx], probs_test))
    try:
        auroc = float(roc_auc_score(labels[test_idx], probs_test))
    except Exception:
        auroc = float('nan')

    thresholds_df = threshold_rates(probs_test, labels[test_idx])

    out = {
        'temperature': t_best,
        'ece': ece,
        'brier': brier,
        'auroc': auroc,
        'n_val': int(n_val),
        'n_test': int(len(test_idx))
    }
    out_path.write_text(json.dumps(out, indent=2))
    thresholds_df.to_csv(out_path.parent / (out_path.stem + '_thresholds.csv'), index=False)
    print('Wrote calibration results to', out_path)


if __name__ == '__main__':
    main()
