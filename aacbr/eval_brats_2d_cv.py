"""
Matched 5-fold CV evaluation for the 2D BraTS-PNG AA-CBR pipeline.

This is the unified, protocol-matched evaluator used for the baselines-vs-AACBR
paper comparison. It runs the same StratifiedKFold(5, shuffle, seed) sweep over
the same characterisation/strategy grid for two feature sources:

  --source gt       features from ground-truth segmentation masks
  --source trained  features from a trained SlotClassifier2D checkpoint

Because the folds depend only on (n_samples, outcomes, seed) 
the GT and trained pipelines are evaluated on exactly
the same patient splits, making their macro-F1 directly comparable. The grid,
run_aacbr strategies and dedup logic are copied directly from
eval_trained_model_brats_2D.ipynb so the trained results already in
leaderboard_trained.json are reproduced by --source trained.
"""

import argparse
import csv as csv_mod
import json
import re
import sys
from itertools import product as iproduct
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

FYP_ROOT = Path(__file__).resolve().parents[1]
if str(FYP_ROOT) not in sys.path:
    sys.path.insert(0, str(FYP_ROOT))

from aacbr.aacbr_parallel import AACBRParallel
from aacbr.configs.brats_model_config import BraTSOutcomeConfig
from datasets.brats2020_png import BraTS2020PNGDataset, batch_seg_to_slot_targets_2d
from utils.characterisations import (
    TumourCharacterisationLarge2D,
    TumourCharacterisationLarge2DV2,
    TumourCharacterisationSmall2D,
    TumourCharacterisationSmall2DV2,
)

CHAR_MODELS = {
    "small":    TumourCharacterisationSmall2D,
    "large":    TumourCharacterisationLarge2D,
    "small_v2": TumourCharacterisationSmall2DV2,
    "large_v2": TumourCharacterisationLarge2DV2,
}


DATA_DIR = str(FYP_ROOT.parent / "Processed_BraTS2020_TrainingData_PNG")
OS_CSV   = str(FYP_ROOT.parent / "BraTS_OS.csv")
CONFIG   = str(FYP_ROOT / "aacbr" / "configs" / "brats_configs" / "flat_config.json")

# Grid. Mirrors eval_trained_model_brats_2D.ipynb but fixes use_supports=False
# (standard AA-CBR with no support cases -- the setting behind 14/16 of the
# trained winners) to keep the search space identical across the GT and trained
# sources while bounding CPU runtime. The same grid is swept for both sources so
# the per-num_classes "best CV-F1" selection is symmetric.
N_BINS_RANGE   = [2, 3, 4, 5]
AGG_MODES      = ["sum", "max"]
STRATEGIES     = ["ordinal", "flat", "tournament_flat", "tournament_tree"]
STRICT_GRID    = [True, False]
SUPPORTS_GRID  = [False]          # set [True, False] to also sweep support cases
SMART_DEF_GRID = [True, False]


# ---------------------------------------------------------------------------
# Outcome loading + patient grouping
# ---------------------------------------------------------------------------

def _parse_survival_days(raw: str) -> float | None:
    raw = raw.strip()
    try:
        return float(raw)
    except ValueError:
        m = re.search(r"(\d+)", raw)
        return float(m.group(1)) if m else None


def load_os_csv(csv_path: str, n_bins: int, resection_filter: str | None = None) -> dict[str, int]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv_mod.DictReader(f):
            pid = row["Brats20ID"].strip()
            days = _parse_survival_days(row["Survival_days"])
            resection = row["Extent_of_Resection"].strip()
            if days is None:
                continue
            if resection_filter and resection != resection_filter:
                continue
            rows.append((pid, days))
    if not rows:
        raise ValueError("No usable rows in OS CSV after filtering.")
    pids = [r[0] for r in rows]
    survival = np.array([r[1] for r in rows])
    quantiles = np.quantile(survival, np.linspace(0, 1, n_bins + 1)[1:-1])
    bins = np.digitize(survival, quantiles)
    return {pid: int(b) for pid, b in zip(pids, bins)}


def group_by_patient(dataset: BraTS2020PNGDataset) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, (t2_path, _) in enumerate(dataset.samples):
        pid = Path(t2_path).parent.name
        groups.setdefault(pid, []).append(i)
    return groups


# ---------------------------------------------------------------------------
# Feature extraction -- GT masks or trained slot model
# ---------------------------------------------------------------------------

def load_slot_model(checkpoint_path: str, device: torch.device):
    from slot_attention.training_2d.slot_attention_2d import SlotClassifier2D

    ckpt = torch.load(checkpoint_path, map_location=device)
    hp = ckpt.get("hyperparameters", {})
    model = SlotClassifier2D(
        in_shape=hp.get("in_shape", (1, 240, 240)),
        width=hp.get("width", 64),
        num_slots=hp.get("num_slots", 5),
        slot_dim=hp.get("slot_dim", 64),
        routing_iters=hp.get("routing_iters", 7),
        temperature=hp.get("temperature", 0.5),
        encoder_depth=hp.get("encoder_depth", 4),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded 2D slot model from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def build_all_features(
    dataset, groups, num_slots, *, source, slot_model=None, device=None,
) -> dict[tuple, dict[str, np.ndarray]]:
    """One pass over all slices -> per-patient features for every (char, agg).

    Slot tuples (seg->targets, or slot-model output) are the expensive part and
    are computed once per slice. All four characterisations and both aggregations
    are accumulated from them. Features are independent of n_bins, so this is
    built once and reused across every binning.
    """
    char_names = list(CHAR_MODELS)
    sum_acc = {c: {} for c in char_names}
    max_acc = {c: {} for c in char_names}
    with torch.no_grad():
        for pid, idxs in tqdm(groups.items(), desc=f"Characterising ({source})"):
            for c in char_names:
                sum_acc[c][pid] = CHAR_MODELS[c].default_case().astype(np.int64)
                max_acc[c][pid] = CHAR_MODELS[c].default_case().astype(np.int64)
            for i in idxs:
                t2, seg = dataset[i]
                if source == "gt":
                    slots = batch_seg_to_slot_targets_2d(seg.unsqueeze(0), num_slots)[0]
                else:
                    _, _, _, _, y_hat = slot_model(t2.unsqueeze(0).to(device))
                    slots = y_hat[0].cpu()
                for c in char_names:
                    v = CHAR_MODELS[c].characterisation_transform(slots).astype(np.int64)
                    sum_acc[c][pid] = sum_acc[c][pid] + v
                    max_acc[c][pid] = np.maximum(max_acc[c][pid], v)
    out: dict[tuple, dict[str, np.ndarray]] = {}
    for c in char_names:
        out[(c, "sum")] = sum_acc[c]
        out[(c, "max")] = max_acc[c]
    return out


# ---------------------------------------------------------------------------
# AA-CBR
# ---------------------------------------------------------------------------

def _dedup(feats: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique, inverse = np.unique(feats, axis=0, return_inverse=True)
    dedup_out = np.array([
        int(np.round(np.median(out[inverse == i]))) for i in range(len(unique))
    ])
    return unique, dedup_out


def run_aacbr(
    train_feats, train_out, test_feats, test_out, char_model, cfg, n_bins,
    *, strategy="ordinal", strict=True, use_supports=False, smart_default=False, dedup=True,
) -> tuple[np.ndarray, np.ndarray]:
    ls_fn = lambda a, b: char_model.less_specific(a, b, strict=strict)

    def make_model(k):
        default_out = (1 if k == 0 else 0) if smart_default else cfg.default_outcome
        return AACBRParallel(
            less_specific=ls_fn,
            default_case=char_model.default_case(),
            default_outcome=default_out,
            include_supports=use_supports,
            supported_attack_chain=use_supports,
        )

    models = {k: make_model(k) for k in range(n_bins)}

    if strategy == "ordinal":
        f, o = _dedup(train_feats, train_out) if dedup else (train_feats, train_out)
        for k in range(n_bins):
            models[k].fit(f, (o >= k).astype(int))
        binary = np.stack([models[k].predict(test_feats) for k in range(n_bins)], axis=1)
        preds = binary[:, 1:].sum(axis=1)

    elif strategy == "flat":
        f, o = _dedup(train_feats, train_out) if dedup else (train_feats, train_out)
        for k in range(n_bins):
            models[k].fit(f, (o == k).astype(int))
        binary = np.stack([models[k].predict(test_feats) for k in range(n_bins)], axis=1)
        preds = np.array([
            max(np.flatnonzero(binary[i] == 1), default=cfg.default_class)
            for i in range(len(test_feats))
        ])

    elif strategy == "tournament_flat":
        for k in range(n_bins - 1):
            mask = train_out >= k
            feats_k, out_k = train_feats[mask], (train_out[mask] == k).astype(int)
            if dedup:
                feats_k, out_k = _dedup(feats_k, out_k)
            models[k].fit(feats_k, out_k)
        preds = np.full(len(test_feats), n_bins - 1, dtype=int)
        unclassified = np.ones(len(test_feats), dtype=bool)
        for k in range(n_bins - 1):
            preds_k = models[k].predict(test_feats[unclassified])
            newly = np.zeros(len(test_feats), dtype=bool)
            newly[unclassified] = preds_k == 1
            preds[newly] = k
            unclassified[newly] = False

    elif strategy == "tournament_tree":
        for k in range(n_bins - 1):
            mask = train_out >= k
            feats_k, out_k = train_feats[mask], (train_out[mask] > k).astype(int)
            if dedup:
                feats_k, out_k = _dedup(feats_k, out_k)
            models[k].fit(feats_k, out_k)
        preds = np.full(len(test_feats), n_bins - 1, dtype=int)
        at_node = np.ones(len(test_feats), dtype=bool)
        for k in range(n_bins - 1):
            preds_k = models[k].predict(test_feats[at_node])
            left_mask = np.zeros(len(test_feats), dtype=bool)
            left_mask[at_node] = preds_k == 0
            preds[left_mask] = k
            at_node[left_mask] = False

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    return test_out, preds


def report(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int) -> dict:
    from sklearn.metrics import confusion_matrix, classification_report
    labels = list(range(n_bins))
    rep = classification_report(y_true, y_pred, labels=labels, zero_division=0,
                                digits=3, output_dict=True)
    macro = rep.get("macro avg", {})
    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1":                float(macro.get("f1-score", 0.0)),
        "precision":         float(macro.get("precision", 0.0)),
        "recall":            float(macro.get("recall", 0.0)),
        "ordinal_mae":       float(np.abs(y_true.astype(float) - y_pred.astype(float)).mean()),
        "confusion_matrix":  confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": {k: {m: v for m, v in vs.items() if m != "support"}
                      for k, vs in rep.items() if isinstance(vs, dict)},
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def _build_feature_cache(source, num_slots, checkpoint=None, resection=None, patient_frac=1.0):
    """Per-(char, n_bins, agg) features. Built once via a single pass over slices."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slot_model = load_slot_model(checkpoint, device) if source == "trained" else None
    dataset = BraTS2020PNGDataset(data_dir=DATA_DIR, is_train=False)
    all_groups = group_by_patient(dataset)
    if patient_frac < 1.0:
        sorted_pids = sorted(all_groups.keys())
        n_keep = max(1, int(len(sorted_pids) * patient_frac))
        keep = set(sorted_pids[:n_keep])
        all_groups = {p: v for p, v in all_groups.items() if p in keep}
    print(f"Source={source} | device={device} | patients={len(all_groups)} | slices={len(dataset)}")
    feats_by_char_agg = build_all_features(
        dataset, all_groups, num_slots, source=source, slot_model=slot_model, device=device)

    feature_cache: dict[tuple, tuple] = {}
    matched: list[str] = []
    for n_bins in N_BINS_RANGE:
        patient_to_class = load_os_csv(OS_CSV, n_bins=n_bins, resection_filter=resection)
        matched = sorted(p for p in all_groups if p in patient_to_class)
        outcomes = np.array([patient_to_class[p] for p in matched])
        for char_name, agg in iproduct(CHAR_MODELS, AGG_MODES):
            feats = feats_by_char_agg[(char_name, agg)]
            feature_cache[(char_name, n_bins, agg)] = (
                np.array([feats[p] for p in matched]), outcomes)
    print(f"Cached {len(feature_cache)} feature sets ({len(matched)} patients).")
    return feature_cache


def _cv_config(feature_cache, kf, cfg, char_name, n_bins, agg,
               strategy, strict, use_supports, smart_default):
    """5-fold CV of one configuration; returns the sweep-result dict."""
    features, outcomes = feature_cache[(char_name, n_bins, agg)]
    char_model = CHAR_MODELS[char_name]
    accs, maes, bals, f1s, precs, recs = [], [], [], [], [], []
    for tr, te in kf.split(features, outcomes):
        yt, yp = run_aacbr(features[tr], outcomes[tr], features[te], outcomes[te],
                           char_model, cfg, n_bins, strategy=strategy, strict=strict,
                           use_supports=use_supports, smart_default=smart_default, dedup=True)
        accs.append(float(accuracy_score(yt, yp)))
        maes.append(float(np.abs(yt.astype(float) - yp.astype(float)).mean()))
        bals.append(float(balanced_accuracy_score(yt, yp)))
        p, r, f, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
        f1s.append(float(f)); precs.append(float(p)); recs.append(float(r))
    return dict(
        char=char_name, n_bins=n_bins, agg=agg, strategy=strategy, strict=strict,
        use_supports=use_supports, smart_default=smart_default,
        mean_acc=float(np.mean(accs)), std_acc=float(np.std(accs)),
        mean_f1=float(np.mean(f1s)), std_f1=float(np.std(f1s)),
        mean_mae=float(np.mean(maes)), std_mae=float(np.std(maes)),
        fold_accs=accs, fold_maes=maes, fold_bal_accs=bals,
        fold_f1s=f1s, fold_precisions=precs, fold_recalls=recs,
    )


def run_sweep(source, n_folds, seed, num_slots, checkpoint=None, resection=None, patient_frac=1.0):
    """Full-grid 5-fold CV over every char/agg/strategy/strict/supports/smart_default."""
    feature_cache = _build_feature_cache(source, num_slots, checkpoint, resection, patient_frac)
    cfg = BraTSOutcomeConfig.from_json(CONFIG)
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    grid = list(iproduct(feature_cache.keys(), STRATEGIES, STRICT_GRID, SUPPORTS_GRID, SMART_DEF_GRID))
    print(f"Running {n_folds}-fold CV over {len(grid)} configurations...")
    sweep = [
        _cv_config(feature_cache, kf, cfg, char_name, n_bins, agg, strategy, strict, sup, smd)
        for (char_name, n_bins, agg), strategy, strict, sup, smd
        in tqdm(grid, desc=f"CV sweep ({source})")
    ]
    sweep.sort(key=lambda r: (-r["mean_acc"], -r["mean_f1"]))
    return sweep, feature_cache, kf, cfg


def configs_from_leaderboard(leaderboard_path, eval_script, checkpoint=None):
    """Best config per (n_bins, strategy) from an existing leaderboard.

    Returns a list of dicts (char/n_bins/agg/strategy/strict/use_supports/
    smart_default) ready for run_configs(). Works for both CV leaderboards
    (metrics carry ``cv_mean_f1``/``oof_f1``) and the single-split GT leaderboard
    (metrics carry ``f1``), so either pipeline's own winning configs can be
    re-evaluated under the matched 5-fold CV without a fresh grid search.

    Pass ``checkpoint`` to restrict to entries from a specific checkpoint path;
    useful when multiple checkpoints coexist in the same leaderboard file.
    """
    data = json.loads(Path(leaderboard_path).read_text())
    best: dict[tuple, tuple] = {}
    for x in data:
        if x.get("eval_script") != eval_script:
            continue
        if checkpoint is not None and x["config"].get("checkpoint") != checkpoint:
            continue
        c, m = x["config"], x["metrics"]
        f1 = m.get("cv_mean_f1") or m.get("oof_f1") or m.get("f1") or 0.0
        key = (c["n_bins"], c["strategy"])
        if key not in best or f1 > best[key][0]:
            best[key] = (f1, dict(
                char=c["char_model"], n_bins=c["n_bins"], agg=c["agg_mode"],
                strategy=c["strategy"], strict=c["strict"],
                use_supports=c.get("use_supports", False),
                smart_default=c.get("smart_default", False)))
    return [v[1] for v in sorted(best.values(), key=lambda kv: (kv[1]["n_bins"], kv[1]["strategy"]))]


def run_configs(source, configs, n_folds, seed, num_slots, checkpoint=None, resection=None, patient_frac=1.0):
    """5-fold CV on an explicit list of configs (from configs_from_leaderboard)."""
    feature_cache = _build_feature_cache(source, num_slots, checkpoint, resection, patient_frac)
    cfg = BraTSOutcomeConfig.from_json(CONFIG)
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    print(f"Running {n_folds}-fold CV over {len(configs)} explicit configs...")
    sweep = [
        _cv_config(feature_cache, kf, cfg, c["char"], c["n_bins"], c["agg"],
                   c["strategy"], c["strict"], c["use_supports"], c["smart_default"])
        for c in tqdm(configs, desc=f"CV configs ({source})")
    ]
    sweep.sort(key=lambda r: (-r["mean_acc"], -r["mean_f1"]))
    return sweep, feature_cache, kf, cfg


def log_best_per_bin_strategy(source, sweep, feature_cache, kf, cfg, n_folds, seed,
                              num_slots, checkpoint, leaderboard_path, notes="", patient_frac=1.0):
    """Mirror the trained-notebook logging: best config per (n_bins, strategy)."""
    from aacbr.results_logger import log_result
    eval_script = "eval_brats_gt_2d_cv" if source == "gt" else "eval_trained_2d_cv"
    for n_bins in sorted({r["n_bins"] for r in sweep}):
        for strategy in STRATEGIES:
            b = next((r for r in sweep if r["strategy"] == strategy and r["n_bins"] == n_bins), None)
            if b is None:
                continue
            features, outcomes = feature_cache[(b["char"], b["n_bins"], b["agg"])]
            dist = np.bincount(outcomes, minlength=b["n_bins"])
            if dist.min() == 0:
                print(f"Skip {strategy} n_bins={n_bins}: degenerate binning {dist.tolist()}")
                continue
            char_model = CHAR_MODELS[b["char"]]
            oof_true, oof_pred = [], []
            for tr, te in kf.split(features, outcomes):
                yt, yp = run_aacbr(features[tr], outcomes[tr], features[te], outcomes[te],
                                   char_model, cfg, b["n_bins"], strategy=b["strategy"],
                                   strict=b["strict"], use_supports=b["use_supports"],
                                   smart_default=b["smart_default"], dedup=True)
                oof_true.extend(yt); oof_pred.extend(yp)
            oof = report(np.array(oof_true), np.array(oof_pred), b["n_bins"])
            log_result({
                "eval_script": eval_script,
                "config": {
                    "checkpoint": checkpoint, "char_model": b["char"], "n_bins": b["n_bins"],
                    "agg_mode": b["agg"], "num_slots": num_slots, "strategy": b["strategy"],
                    "strict": b["strict"], "use_supports": b["use_supports"],
                    "smart_default": b["smart_default"], "n_folds": n_folds, "seed": seed,
                    "source": source, "patient_frac": patient_frac,
                },
                "data_stats": {"n_patients": int(len(outcomes)), "class_distribution": dist.tolist()},
                "metrics": {
                    "cv_mean_accuracy": b["mean_acc"], "cv_std_accuracy": b["std_acc"],
                    "cv_mean_balanced_accuracy": float(np.mean(b["fold_bal_accs"])),
                    "cv_std_balanced_accuracy": float(np.std(b["fold_bal_accs"])),
                    "cv_mean_f1": b["mean_f1"], "cv_std_f1": b["std_f1"],
                    "cv_mean_precision": float(np.mean(b["fold_precisions"])),
                    "cv_std_precision": float(np.std(b["fold_precisions"])),
                    "cv_mean_recall": float(np.mean(b["fold_recalls"])),
                    "cv_std_recall": float(np.std(b["fold_recalls"])),
                    "cv_mean_mae": b["mean_mae"], "cv_std_mae": b["std_mae"],
                    "fold_accs": b["fold_accs"], "fold_maes": b["fold_maes"],
                    "fold_balanced_accs": b["fold_bal_accs"], "fold_f1s": b["fold_f1s"],
                    "oof_accuracy": oof["accuracy"], "oof_balanced_accuracy": oof["balanced_accuracy"],
                    "oof_f1": oof["f1"], "oof_precision": oof["precision"], "oof_recall": oof["recall"],
                    "oof_ordinal_mae": oof["ordinal_mae"], "confusion_matrix": oof["confusion_matrix"],
                    "per_class": oof["per_class"],
                },
                "notes": notes,
            }, leaderboard_path=leaderboard_path)
            print(f"  logged {eval_script} n_bins={n_bins} {strategy}: "
                  f"cv_f1={b['mean_f1']:.4f}±{b['std_f1']:.4f} (char={b['char']}, agg={b['agg']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, choices=["gt", "trained"])
    ap.add_argument("--checkpoint", default=None, help="[trained] SlotClassifier2D .pt")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_slots", type=int, default=5)
    ap.add_argument("--resection", default=None)
    ap.add_argument("--save", action="store_true", help="Append best-per-(n_bins,strategy) to the leaderboard.")
    ap.add_argument("--leaderboard", default=None,
                    help="Default: leaderboard_gt_cv.json or leaderboard_trained_cv.json.")
    ap.add_argument("--configs-from", default=None,
                    help="Re-use the best config per (n_bins, strategy) from this leaderboard "
                         "instead of a full grid search (much faster).")
    ap.add_argument("--configs-eval-script", default="eval_trained_2d",
                    help="eval_script tag to read configs from with --configs-from.")
    ap.add_argument("--patient-frac", type=float, default=1.0,
                    help="Fraction of patients to use (first N%% alphabetically, "
                         "matching eval_2d_slot.ipynb subset logic). Default: 1.0 (all).")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    if args.source == "trained" and not args.checkpoint:
        ap.error("--checkpoint is required for --source trained")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.configs_from:
        configs = configs_from_leaderboard(args.configs_from, args.configs_eval_script,
                                           checkpoint=args.checkpoint)
        print(f"Loaded {len(configs)} configs from {args.configs_from} "
              f"(eval_script={args.configs_eval_script})")
        sweep, feature_cache, kf, cfg = run_configs(
            args.source, configs, args.n_folds, args.seed, args.num_slots,
            checkpoint=args.checkpoint, resection=args.resection,
            patient_frac=args.patient_frac)
    else:
        sweep, feature_cache, kf, cfg = run_sweep(
            args.source, args.n_folds, args.seed, args.num_slots,
            checkpoint=args.checkpoint, resection=args.resection,
            patient_frac=args.patient_frac)

    # Console summary: best CV-F1 config per n_bins (the paper selection rule).
    print("\nBest CV-F1 config per n_bins:")
    print(f"{'bins':>4} {'char':<9} {'agg':<4} {'strategy':<16} {'cv_f1':>14}  {'cv_acc':>14}")
    for n_bins in sorted({r["n_bins"] for r in sweep}):
        b = max((r for r in sweep if r["n_bins"] == n_bins), key=lambda r: r["mean_f1"])
        print(f"{n_bins:>4} {b['char']:<9} {b['agg']:<4} {b['strategy']:<16} "
              f"{b['mean_f1']:>7.4f}±{b['std_f1']:.3f}  {b['mean_acc']:>7.4f}±{b['std_acc']:.3f}")

    if args.save:
        lb = args.leaderboard or str(
            FYP_ROOT / "aacbr" / (f"leaderboard_gt_cv.json" if args.source == "gt"
                                  else "leaderboard_trained_cv.json"))
        log_best_per_bin_strategy(args.source, sweep, feature_cache, kf, cfg,
                                  args.n_folds, args.seed, args.num_slots,
                                  args.checkpoint, lb, notes=args.notes,
                                  patient_frac=args.patient_frac)
        print(f"\nLeaderboard written to {lb}")


if __name__ == "__main__":
    main()
