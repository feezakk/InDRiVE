# tools/plot_eval.py
import json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_df(run):
    with open(Path(run) / "eval_dump" / "episodes.jsonl") as f:
        rows = [json.loads(x) for x in f]
    return pd.DataFrame(rows)

def load_npzs(run):
    d = Path(run) / "eval_dump"
    for npz in sorted(d.glob("ep_*.npz")):
        yield np.load(npz, allow_pickle=False)

def plot_reliability(run, label, outdir):
    # ANY raw vs cal (aggregated across episodes by count-weighted bin means)
    df = load_df(run)
    bins  = np.array(df.iloc[0]["raw_any_bins"], float)
    acc_r = np.zeros_like(bins); conf_r = np.zeros_like(bins); cnt_r = np.zeros_like(bins)
    acc_c = np.zeros_like(bins); conf_c = np.zeros_like(bins); cnt_c = np.zeros_like(bins)
    for _, r in df.iterrows():
        acc_r += np.array(r["raw_any_acc"]) * np.array(r["raw_any_cnt"])
        conf_r += np.array(r["raw_any_conf"]) * np.array(r["raw_any_cnt"])
        cnt_r += np.array(r["raw_any_cnt"])
        acc_c += np.array(r["cal_any_acc"]) * np.array(r["cal_any_cnt"])
        conf_c += np.array(r["cal_any_conf"]) * np.array(r["cal_any_cnt"])
        cnt_c += np.array(r["cal_any_cnt"])
    acc_r = np.divide(acc_r, np.maximum(cnt_r, 1), out=np.zeros_like(acc_r), where=cnt_r>0)
    conf_r = np.divide(conf_r, np.maximum(cnt_r, 1), out=np.zeros_like(conf_r), where=cnt_r>0)
    acc_c = np.divide(acc_c, np.maximum(cnt_c, 1), out=np.zeros_like(acc_c), where=cnt_c>0)
    conf_c = np.divide(conf_c, np.maximum(cnt_c, 1), out=np.zeros_like(conf_c), where=cnt_c>0)

    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1], linestyle="--")
    plt.plot(conf_r, acc_r, marker="o", label=f"{label} (raw)")
    plt.plot(conf_c, acc_c, marker="o", label=f"{label} (cal)")
    plt.xlabel("Confidence"); plt.ylabel("Empirical frequency"); plt.title("Reliability (ANY)")
    plt.legend(); plt.tight_layout()
    plt.savefig(Path(outdir) / f"fig_reliability_{Path(run).name}.png", dpi=200); plt.close()

def plot_risk_coverage(run, label, outdir):
    cov_r_all, risk_r_all, cov_c_all, risk_c_all = [], [], [], []
    for npz in load_npzs(run):
        cov_r_all.append(npz["cov_raw"]); risk_r_all.append(npz["risk_raw"])
        cov_c_all.append(npz["cov_cal"]); risk_c_all.append(npz["risk_cal"])
    # average on a common grid
    grid = np.linspace(0, 1, 101)
    def interp(xs, ys):
        arr = []
        for x,y in zip(xs, ys):
            arr.append(np.interp(grid, x, y))
        return np.vstack(arr).mean(0)
    r_raw = interp(cov_r_all, risk_r_all)
    r_cal = interp(cov_c_all, risk_c_all)
    aurc_raw = np.trapz(r_raw, grid); aurc_cal = np.trapz(r_cal, grid)

    plt.figure(figsize=(5,4))
    plt.plot(grid, r_raw, label=f"{label} raw (AURC={aurc_raw:.3f})")
    plt.plot(grid, r_cal, label=f"{label} cal (AURC={aurc_cal:.3f})")
    plt.xlabel("Coverage"); plt.ylabel("Risk among kept"); plt.title("Risk–Coverage (ANY)")
    plt.legend(); plt.tight_layout()
    plt.savefig(Path(outdir) / f"fig_risk_coverage_{Path(run).name}.png", dpi=200); plt.close()

def plot_lead_hist(run, label, outdir):
    df = load_df(run)
    leads = []
    for npz in load_npzs(run):
        y = npz["y_any"].astype(bool); q = npz["q_any"]; tau = float(npz["tau"])
        idxs = np.where(y)[0]
        for t in idxs:
            prev = np.where(q[:t+1] >= tau)[0]
            if prev.size: leads.append((t - prev[-1]) * float(npz["dt"]))
    if not leads: leads = [0.0]
    plt.figure(figsize=(5,4))
    plt.hist(leads, bins=30)
    plt.xlabel("Lead time (s)"); plt.ylabel("Count")
    plt.title(f"Lead‑time histogram — {label}")
    plt.tight_layout()
    plt.savefig(Path(outdir) / f"fig_lead_hist_{Path(run).name}.png", dpi=200); plt.close()

def plot_ablation_bars(run_dirs, labels, outdir):
    rows = []
    for run, lab in zip(run_dirs, labels):
        df = load_df(run)
        rows.append({"method": lab,
                     "hazards/km": df["hazards_per_km"].mean(),
                     "overrides_%": df["overrides_pct"].mean()})
    T = pd.DataFrame(rows)
    # bars
    plt.figure(figsize=(6,4))
    x = np.arange(len(T))
    w = 0.4
    plt.bar(x - w/2, T["hazards/km"], width=w, label="hazards/km")
    plt.bar(x + w/2, T["overrides_%"], width=w, label="overrides %")
    plt.xticks(x, T["method"], rotation=15)
    plt.legend(); plt.tight_layout()
    plt.savefig(Path(outdir) / "fig_ablation_bars.png", dpi=200); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--out", default="eval_figs")
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    labels = args.labels if args.labels else [Path(d).name for d in args.run_dirs]

    for run, lab in zip(args.run_dirs, labels):
        plot_reliability(run, lab, outdir)
        plot_risk_coverage(run, lab, outdir)
        plot_lead_hist(run, lab, outdir)
    if len(args.run_dirs) > 1:
        plot_ablation_bars(args.run_dirs, labels, outdir)
    print(f"Saved figures to {outdir}")

if __name__ == "__main__":
    main()
