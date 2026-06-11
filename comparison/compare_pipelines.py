"""Three-way performance comparison for the paper.

Compares, as a function of the number of output classes k in {2,3,4,5}:

  * Baseline        -- best end-to-end / AE-classifier ResNet (5-fold OOF macro-F1)
  * Slot+AA-CBR     -- trained SlotClassifier2D features + AA-CBR (5-fold CV macro-F1)
  * GT+AA-CBR       -- ground-truth segmentation features + AA-CBR (5-fold CV macro-F1)

All three are evaluated under matched 5-fold cross-validation on the same
patients, so macro-F1 is directly comparable within each k. The two AA-CBR rows
report the best-CV-F1 characterisation/strategy configuration per k (selection
rule disclosed in the table footnote).

The metric (macro-F1) is not comparable *across* k -- a k-class problem has a
lower chance level -- so every figure/table also shows the chance line 1/k and a
"skill" view (macro-F1 minus chance) to keep the trend interpretable.

Outputs (under comparison/):
  figures/compare_<dataset>.png
  tables/compare_<dataset>.tex
  tables/compare_<dataset>.csv
  results/compare_<dataset>.json

    python -m comparison.compare_pipelines --dataset brats_png
"""

import argparse
import json
from pathlib import Path

import numpy as np

FYP_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = FYP_ROOT / "comparison"
NUM_CLASSES = [2, 3, 4, 5]

NUM_CLASSES_LB = FYP_ROOT / "baseline" / "num_classes_leaderboard.json"

DATASETS = {
    "brats_png": {
        "title": "BraTS: overall-survival classification",
        "baseline_dataset": "brats_png",
        "baseline_pipelines": ["end_to_end_2d", "ae_classifier_2d"],
        # Baseline is trained on the 85% training patients and evaluated on the
        # same 15% held-out patients as Slot/GT+AA-CBR (holdout_frac=0.15).
        "baseline_holdout_frac": 0.15,
        # Both GT and Slot+AA-CBR are evaluated under matched 5-fold CV from
        # run_comparison.ipynb, which writes leaderboard_*_cv.json.
        "gt_leaderboard": FYP_ROOT / "aacbr" / "leaderboard_gt_cv.json",
        "gt_eval_script": "eval_brats_gt_2d_cv",
        "trained_leaderboard": FYP_ROOT / "aacbr" / "leaderboard_trained_cv.json",
        "trained_eval_script": "eval_trained_2d_cv",
    },
}

# Display order + style for the canonical methods.
METHODS = ["Baseline", "Slot+AA-CBR", "GT+AA-CBR"]
STYLE = {
    "Baseline":    dict(color="#4C72B0", marker="o"),
    "Slot+AA-CBR": dict(color="#DD8452", marker="s"),
    "GT+AA-CBR":   dict(color="#55A868", marker="^"),
}

# Extra styles cycled over when there are multiple Slot+AA-CBR variants.
_SLOT_STYLES = [
    dict(color="#DD8452", marker="s"),  # orange  (matches original)
    dict(color="#C44E52", marker="D"),  # red
    dict(color="#8172B2", marker="P"),  # purple
    dict(color="#937860", marker="X"),  # brown
]


def _build_styles(methods: list[str]) -> dict[str, dict]:
    """Return a style dict for each method, cycling _SLOT_STYLES for unknown names."""
    styles: dict[str, dict] = {}
    slot_idx = 0
    for m in methods:
        if m in STYLE:
            styles[m] = STYLE[m]
        else:
            styles[m] = _SLOT_STYLES[slot_idx % len(_SLOT_STYLES)]
            slot_idx += 1
    return styles


def _load_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def chance_level(n_bins: int) -> float:
    """Macro-F1 of uniform-random guessing under (near-)balanced quantile bins."""
    return 1.0 / n_bins


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_baseline(ds_cfg) -> dict[int, dict]:
    lb = _load_json(NUM_CLASSES_LB)
    out: dict[int, dict] = {}
    if lb is None:
        return out
    holdout_frac = ds_cfg.get("baseline_holdout_frac")
    rows = [r for r in lb.get("ranked", [])
            if r.get("dataset") == ds_cfg["baseline_dataset"]
            and r.get("pipeline") in ds_cfg["baseline_pipelines"]]
    if holdout_frac is not None:
        # Use holdout test-set F1: baseline trained on 85%, evaluated on held-out 15%.
        rows = [r for r in rows
                if r.get("holdout_frac") == holdout_frac
                and r.get("holdout_cv_mean_f1") is not None]
        for nc in NUM_CLASSES:
            cands = [r for r in rows if r.get("num_classes") == nc]
            if not cands:
                continue
            best = max(cands, key=lambda r: r["holdout_cv_mean_f1"])
            out[nc] = {
                "f1": float(best["holdout_cv_mean_f1"]),
                "std": float(best["holdout_cv_std_f1"]) if best.get("holdout_cv_std_f1") is not None else None,
                "config": f"{best.get('pipeline')} ({best.get('model')})",
            }
    else:
        rows = [r for r in rows if r.get("f1") is not None]
        for nc in NUM_CLASSES:
            cands = [r for r in rows if r.get("num_classes") == nc]
            if not cands:
                continue
            best = max(cands, key=lambda r: r["f1"])
            std = None
            res = _load_json(Path(best["path"]) / "results.json") if best.get("path") else None
            if res:
                folds = [fm["f1"] for fm in res.get("fold_metrics", []) if fm.get("f1") is not None]
                if folds:
                    std = float(np.std(folds))
            out[nc] = {"f1": float(best["f1"]), "std": std,
                       "config": f"{best.get('pipeline')} ({best.get('model')})"}
    return out


def load_aacbr(lb_path, eval_script, checkpoint=None) -> dict[int, dict]:
    d = _load_json(lb_path)
    out: dict[int, dict] = {}
    if d is None:
        return out
    rows = [x for x in d if x.get("eval_script") == eval_script
            and x.get("metrics", {}).get("cv_mean_f1") is not None]
    if checkpoint is not None:
        rows = [x for x in rows if x.get("config", {}).get("checkpoint") == checkpoint]
    for nc in NUM_CLASSES:
        cands = [x for x in rows if x["config"].get("n_bins") == nc]
        if not cands:
            continue
        best = max(cands, key=lambda x: x["metrics"]["cv_mean_f1"])
        c, m = best["config"], best["metrics"]
        out[nc] = {"f1": float(m["cv_mean_f1"]), "std": float(m.get("cv_std_f1") or 0.0),
                   "config": f"{c.get('char_model')}/{c.get('agg_mode')}/{c.get('strategy')}"}
    return out


def assemble(
    ds_cfg,
    checkpoint_labels: dict[str, str] | None = None,
    only_checkpoints: list[str] | None = None,
) -> dict[str, dict[int, dict]]:
    """Build the unified data dict for comparison.

    Args:
        ds_cfg: dataset config from DATASETS.
        checkpoint_labels: optional dict mapping checkpoint folder name (e.g.
            ``"brats_png_v14a_gamma4"``) to a custom display label used inside
            the brackets, e.g. ``{"brats_png_v14a_gamma4": "v14a (γ=4)"}``.
            Falls back to the folder name when a checkpoint is not listed.
        only_checkpoints: optional list of checkpoint folder names to include
            (e.g. ``["brats_png_v14a_gamma4", "brats_png_v20a_attn_skip"]``).
            When ``None`` all checkpoints found in the leaderboard are used.
    """
    lb_path = ds_cfg["trained_leaderboard"]
    eval_script = ds_cfg["trained_eval_script"]

    # Auto-detect unique checkpoints in the leaderboard.
    raw = _load_json(lb_path) or []
    ckpts = sorted({
        x["config"]["checkpoint"]
        for x in raw
        if x.get("eval_script") == eval_script and x.get("config", {}).get("checkpoint")
    })
    if only_checkpoints is not None:
        ckpts = [c for c in ckpts if Path(c).parent.name in only_checkpoints]

    labels = checkpoint_labels or {}
    result: dict[str, dict[int, dict]] = {"Baseline": load_baseline(ds_cfg)}
    if len(ckpts) <= 1:
        result["Slot+AA-CBR"] = load_aacbr(lb_path, eval_script,
                                            checkpoint=ckpts[0] if ckpts else None)
    else:
        for ckpt in ckpts:
            short = Path(ckpt).parent.name
            display = labels.get(short, short)
            result[f"Slot+AA-CBR ({display})"] = load_aacbr(lb_path, eval_script, checkpoint=ckpt)
    result["GT+AA-CBR"] = load_aacbr(ds_cfg["gt_leaderboard"], ds_cfg["gt_eval_script"])
    return result


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(data, title, out_stub):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(data.keys())
    styles = _build_styles(methods)

    # Figure A: raw macro-F1.
    figA, axL = plt.subplots(figsize=(5.5, 4.2))
    for name in methods:
        d = data[name]
        ks = [k for k in NUM_CLASSES if k in d]
        if not ks:
            continue
        ys = [d[k]["f1"] for k in ks]
        axL.plot(ks, ys, lw=1.8, markersize=6, label=name, zorder=3, **styles[name])
    axL.set_xlabel("Number of output classes $k$")
    axL.set_ylabel("Macro-F1 (5-fold)")
    axL.set_title("Macro-F1 vs. class granularity")
    axL.set_xticks(NUM_CLASSES)
    axL.grid(True, ls=":", alpha=0.5)
    axL.legend(fontsize=8, framealpha=0.9)
    figA.suptitle(title, fontsize=12, y=1.02)
    figA.tight_layout()
    figA.savefig(f"{out_stub}_f1.png", bbox_inches="tight", dpi=200)
    plt.close(figA)
    print(f"figure -> {out_stub}_f1.png")

    # Figure B: skill = macro-F1 above chance (comparable across k).
    figB, axR = plt.subplots(figsize=(5.5, 4.2))
    for name in methods:
        d = data[name]
        ks = [k for k in NUM_CLASSES if k in d]
        if not ks:
            continue
        ys = [d[k]["f1"] - chance_level(k) for k in ks]
        axR.plot(ks, ys, lw=1.8, markersize=6, label=name, zorder=3, **styles[name])
    axR.set_xlabel("Number of output classes $k$")
    axR.set_ylabel("Macro-F1 $-$ chance")
    axR.set_title("Performance above chance")
    axR.set_xticks(NUM_CLASSES)
    axR.grid(True, ls=":", alpha=0.5)
    axR.legend(fontsize=8, framealpha=0.9)
    figB.suptitle(title, fontsize=12, y=1.02)
    figB.tight_layout()
    figB.savefig(f"{out_stub}_skill.png", bbox_inches="tight", dpi=200)
    plt.close(figB)
    print(f"figure -> {out_stub}_skill.png")


# ---------------------------------------------------------------------------
# Tabular views (rendered DataFrame + CSV)
# ---------------------------------------------------------------------------

def to_dataframe(data):
    """Numeric macro-F1 table: methods (+ chance) as rows, k as columns."""
    import pandas as pd
    cols = [f"k={k}" for k in NUM_CLASSES]
    idx, rows = [], []
    for name in data:
        idx.append(name)
        rows.append([data[name].get(k, {}).get("f1", np.nan) for k in NUM_CLASSES])
    idx.append("Chance (1/k)")
    rows.append([chance_level(k) for k in NUM_CLASSES])
    return pd.DataFrame(rows, index=idx, columns=cols)


def config_dataframe(data):
    """Best-CV-F1 AA-CBR configuration (char/agg/strategy) per k."""
    import pandas as pd
    cols = [f"k={k}" for k in NUM_CLASSES]
    idx, rows = [], []
    for name in [m for m in data if m != "Baseline"]:
        idx.append(name)
        rows.append([data[name].get(k, {}).get("config", "--") for k in NUM_CLASSES])
    return pd.DataFrame(rows, index=idx, columns=cols)


def write_csv(data, out_path):
    """Long-format CSV: one row per (method, k) with f1, std, config, chance."""
    import csv as _csv
    with open(out_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["method", "k", "macro_f1", "std", "config", "chance"])
        for name in data:
            for k in NUM_CLASSES:
                d = data[name].get(k)
                if not d:
                    continue
                w.writerow([name, k, f"{d['f1']:.4f}",
                            "" if d.get("std") is None else f"{d['std']:.4f}",
                            d.get("config", ""), f"{chance_level(k):.4f}"])
    print(f"csv    -> {out_path}")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def _fmt(d):
    if d is None:
        return "--"
    s = d["f1"]
    return f"{s:.3f}" + (f"\\,$\\pm$\\,{d['std']:.3f}" if d.get("std") else "")


def make_table(data, dataset, title, out_path):
    methods = list(data.keys())
    head = " & ".join(f"$k={k}$" for k in NUM_CLASSES)
    lines = [
        "% Auto-generated by comparison/compare_pipelines.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{" + title + ". Macro-F1 (mean over 5 folds $\\pm$ standard "
        "deviation) versus the number of output classes $k$. All methods share the "
        "same patients and 5-fold split. \\textbf{Bold} = best per column. The "
        "AA-CBR rows report the best-CV-F1 characterisation/aggregation/strategy "
        "configuration per $k$ (listed beneath). Macro-F1 is comparable down each "
        "column but not across columns; the chance row ($1/k$) and "
        "Fig.~\\ref{fig:compare-" + dataset + "} give the cross-$k$ context.}",
        "\\label{tab:compare-" + dataset + "}",
        "\\begin{tabular}{l" + "c" * len(NUM_CLASSES) + "}",
        "\\toprule",
        "Method & " + head + " \\\\",
        "\\midrule",
    ]
    # bold best per column
    best_per_k = {}
    for k in NUM_CLASSES:
        vals = {m: data[m].get(k, {}).get("f1") for m in methods}
        vals = {m: v for m, v in vals.items() if v is not None}
        if vals:
            best_per_k[k] = max(vals, key=vals.get)
    for name in methods:
        cells = []
        for k in NUM_CLASSES:
            d = data[name].get(k)
            txt = _fmt(d)
            if d is not None and best_per_k.get(k) == name:
                txt = f"\\textbf{{{txt}}}"
            cells.append(txt)
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    chance_cells = " & ".join(f"{chance_level(k):.3f}" for k in NUM_CLASSES)
    lines += [
        "\\midrule",
        f"\\textit{{Chance}} ($1/k$) & {chance_cells} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    # config footnotes for all non-Baseline AA-CBR rows
    notes = []
    for name in [m for m in methods if m != "Baseline"]:
        cfgs = "; ".join(f"$k{k}$: {data[name][k]['config']}"
                         for k in NUM_CLASSES if k in data[name])
        if cfgs:
            notes.append(f"\\footnotesize {name} config (char/agg/strategy) -- {cfgs}.")
    if notes:
        lines.append("\\\\[2pt]")
        lines.append(" \\\\ ".join(notes))
    lines.append("\\end{table}")
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"table  -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="brats_png", choices=list(DATASETS))
    args = ap.parse_args()
    ds_cfg = DATASETS[args.dataset]
    title = ds_cfg["title"]

    data = assemble(ds_cfg)
    for name in data:
        present = sorted(data[name])
        print(f"{name:30s}: k={present or 'none'}")

    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results").mkdir(parents=True, exist_ok=True)

    make_figure(data, title, OUT_DIR / "figures" / f"compare_{args.dataset}")
    make_table(data, args.dataset, title, OUT_DIR / "tables" / f"compare_{args.dataset}.tex")
    write_csv(data, OUT_DIR / "tables" / f"compare_{args.dataset}.csv")
    (OUT_DIR / "results" / f"compare_{args.dataset}.json").write_text(
        json.dumps({"title": title, "num_classes": NUM_CLASSES, "data": data}, indent=2))
    print(f"results -> {OUT_DIR / 'results' / f'compare_{args.dataset}.json'}")

    try:  # console preview of the F1 comparison table
        import pandas as pd
        with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
            print("\nMacro-F1 comparison:\n", to_dataframe(data).to_string(na_rep="--"), sep="")
    except Exception:
        pass


if __name__ == "__main__":
    main()
