#!/usr/bin/env python3
"""
Parse evaluation CSVs produced by `tools/log_eval_episode.py` and summarize shield metrics.

Usage:
  python tools/parse_eval_results.py --dir ./eval_results

It groups files by a lambda value inferred from the filename (look for 'lam' followed by digits/decimal, e.g. 'lam0.15').
If no lam is found in a filename, files are grouped under 'unknown'.
"""
import argparse
import csv
import os
import re
import ast
from collections import defaultdict


def parse_action_field(s):
    # action may be a Python list representation or a scalar
    if s is None:
        return None
    s = s.strip()
    # try literal_eval
    try:
        v = ast.literal_eval(s)
        return v
    except Exception:
        # fallback: if comma separated
        if ',' in s:
            try:
                parts = [float(x) for x in s.split(',')]
                return parts
            except Exception:
                return s
        try:
            return float(s)
        except Exception:
            return s


def detect_lam(filename):
    m = re.search(r"lam\s*=?\s*([0-9]+(?:\.[0-9]+)?)", filename, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m2 = re.search(r"_l?([0-9]+(?:_[0-9]+)?)", filename)
    if m2:
        # e.g. _015 -> 0.15 or _0 -> 0
        token = m2.group(1)
        if '_' in token:
            token = token.replace('_', '.')
        try:
            v = float(token)
            # heuristic: if token looks like int 15 and filename contains 'lam015', treat as 0.15 when length==3
            if v > 1 and len(token) <= 3:
                return v / 100.0
            return v
        except Exception:
            return None
    return None


def summarize_file(path):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return None
    n = len(rows)
    shield_count = 0
    speeds = []
    rewards = []
    collisions = 0
    for r in rows:
        s = r.get('shield_unsafe') or r.get('log_shield_unsafe')
        try:
            if s is not None and str(s).strip() != '' and float(s) > 0:
                shield_count += 1
        except Exception:
            # if it's a boolean-like
            if str(s).lower() in ('true','1'):
                shield_count += 1
        sk = r.get('speed_kmh')
        try:
            if sk is not None and sk != '':
                speeds.append(float(sk))
        except Exception:
            pass
        rew = r.get('reward')
        try:
            if rew is not None and rew != '':
                rewards.append(float(rew))
        except Exception:
            pass
        # collision field may be present
        c = r.get('is_collision') or r.get('collision')
        try:
            if c is not None and c != '' and float(c) > 0:
                collisions += 1
        except Exception:
            if str(c).lower() in ('true','1'):
                collisions += 1

    return {
        'steps': n,
        'shield_count': shield_count,
        'override_rate': shield_count / n if n else 0.0,
        'mean_speed_kmh': sum(speeds) / len(speeds) if speeds else None,
        'total_reward': sum(rewards) if rewards else None,
        'mean_reward': sum(rewards) / len(rewards) if rewards else None,
        'collisions': collisions,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', required=True, help='Directory with CSV files to parse')
    args = p.parse_args()

    files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.lower().endswith('.csv')]
    if not files:
        print('No CSV files found in', args.dir)
        return

    by_lam = defaultdict(list)
    per_file = {}
    for f in sorted(files):
        lam = detect_lam(os.path.basename(f))
        lam_key = lam if lam is not None else 'unknown'
        s = summarize_file(f)
        per_file[f] = (lam_key, s)
        by_lam[lam_key].append((f, s))

    print('Per-file summaries:')
    for f, (lam_key, s) in per_file.items():
        print('-', os.path.basename(f), 'lam=', lam_key, s)

    print('\nAggregated by lambda:')
    for lam_key, lst in sorted(by_lam.items(), key=lambda x: (str(x[0]))):
        total_steps = sum(s['steps'] for _, s in lst if s)
        total_shields = sum(s['shield_count'] for _, s in lst if s)
        mean_speed = None
        speeds = [s['mean_speed_kmh'] for _, s in lst if s and s['mean_speed_kmh'] is not None]
        rewards = [s['mean_reward'] for _, s in lst if s and s['mean_reward'] is not None]
        collisions = sum(s['collisions'] for _, s in lst if s)
        if speeds:
            mean_speed = sum(speeds) / len(speeds)
        mean_reward = sum(rewards) / len(rewards) if rewards else None
        override_rate = total_shields / total_steps if total_steps else 0.0
        print(f"lam={lam_key}: episodes={len(lst)}, steps={total_steps}, override_rate={override_rate:.3f}, mean_speed_kmh={mean_speed}, mean_reward={mean_reward}, collisions={collisions}")


if __name__ == '__main__':
    main()
