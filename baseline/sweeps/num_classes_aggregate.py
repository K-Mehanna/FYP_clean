"""Aggregate num-classes sweep results into num_classes_leaderboard.json.

Scans every dataset under each pipeline for runs that belong to the controlled
num_classes sweep (num_classes in [2, 3, 4, 5], best hyperparameters fixed).
End-to-end datasets are discovered dynamically, so brats_png / brats / future
datasets are included automatically.

Identification strategy per pipeline family:
  end_to_end / end_to_end_2d  — require ``num_classes_sweep: true`` in args.json
                                 (set by run_num_classes_sweep.ipynb; these runs
                                 also have a _nb{N} suffix in their directory name)
  ae_classifier / ae_classifier_2d — match runs whose non-num_classes
                                 hyperparameters equal the best config from the
                                 main leaderboard.  This naturally includes the
                                 pre-existing nc=3 run that predates the sweep
                                 flag, as well as the new nc=[2,4,5] runs.

    python -m baseline.sweeps.num_classes_aggregate
"""

import json
import sys
from datetime import date
from pathlib import Path

FYP_ROOT = Path(__file__).resolve().parents[2]
if str(FYP_ROOT) not in sys.path:
    sys.path.insert(0, str(FYP_ROOT))

BASELINE_DIR          = FYP_ROOT / "baseline"
END_TO_END_RUNS       = BASELINE_DIR / "end-to-end" / "runs"
AE_CLASSIFIER_RUNS    = BASELINE_DIR / "autoencoder" / "models" / "classifier"
AE_CLASSIFIER_2D_RUNS = BASELINE_DIR / "autoencoder" / "models" / "classifier_2d"
MAIN_LEADERBOARD_PATH = BASELINE_DIR / "leaderboard.json"
LEADERBOARD_PATH      = BASELINE_DIR / "num_classes_leaderboard.json"

NUM_CLASSES_RANGE = {2, 3, 4, 5}
PRIMARY_METRIC = "balanced_accuracy"
CAVEAT = (
    "balanced_accuracy across num_classes=[2,3,4,5] with best hyperparameters fixed per pipeline. "
    "Compare within a pipeline to see how granularity affects performance."
)

# Hyperparameter keys used to identify whether a run matches the best config
# (num_classes is excluded because that is what we're sweeping).
_AE_MATCH_KEYS  = ("ae_weights", "lr", "dropout", "weight_decay", "epochs", "folds", "quantile")
_E2E_MATCH_KEYS = ("model", "lr", "weight_decay", "epochs", "n_splits", "quantile", "strategy")


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print("  [warn] could not read {}: {}".format(path, exc))
        return None


def _macro_metrics(oof):
    macro = oof.get("classification_report", {}).get("macro avg", {})
    def _get(key, report_key):
        v = oof.get(key)
        if v is not None:
            return float(v)
        return float(macro[report_key]) if report_key in macro else None
    return {
        "precision": _get("precision", "precision"),
        "recall":    _get("recall",    "recall"),
        "f1":        _get("f1",        "f1-score"),
    }


def _best_configs_from_leaderboard():
    """Return {(pipeline, dataset): config_dict} for the top F1 hyperparameter-sweep run per group.

    Only considers runs from the hyperparameter sweep (score_name is None or 'QL2').
    Outcome-sweep runs (BNCD, FA, etc.) are excluded so that the selected config
    reflects the best general hyperparameters, not a target-specific tuning.
    """
    data = _load_json(MAIN_LEADERBOARD_PATH)
    if data is None:
        return {}
    seen = {}
    for r in data.get("ranked", []):
        if r.get("needs_rerun") or r.get("f1") is None:
            continue
        # Skip outcome-specific runs; keep only the base hyperparameter sweep runs.
        if r.get("score_name") not in (None, "QL2"):
            continue
        key = (r["pipeline"], r.get("dataset", "brainwear"))
        if key not in seen:
            seen[key] = r["config"]
    return seen


def _configs_match(saved_args, best_cfg, match_keys):
    for k in match_keys:
        if saved_args.get(k) != best_cfg.get(k):
            return False
    return True


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

def _collect_end_to_end(dataset_dir, pipeline, best_cfg):
    records = []
    runs_dir = END_TO_END_RUNS / dataset_dir
    if not runs_dir.exists():
        return records
    for results_path in sorted(runs_dir.glob("*/results.json")):
        results = _load_json(results_path)
        if results is None:
            continue
        run_dir = results_path.parent
        args = _load_json(run_dir / "args.json") or {}
        if not args.get("num_classes_sweep"):
            continue
        nc = args.get("num_bins")
        if nc not in NUM_CLASSES_RANGE:
            continue
        if best_cfg and not _configs_match(args, best_cfg, _E2E_MATCH_KEYS):
            continue
        summary = results.get("summary", {})
        oof = results.get("oof_results", {})
        records.append({
            "pipeline": pipeline,
            "dataset": dataset_dir,
            "run_name": run_dir.name,
            "model": summary.get("model") or args.get("model"),
            "num_classes": nc,
            "balanced_accuracy": oof.get("balanced_accuracy"),
            "accuracy": summary.get("accuracy_mean"),
            "val_loss": summary.get("best_val_loss_mean"),
            "config": {k: args.get(k) for k in ("model", "lr", "weight_decay", "epochs",
                                                  "n_splits", "num_bins", "quantile", "strategy")},
            "path": str(run_dir),
            "holdout_frac": summary.get("holdout_frac"),
            "holdout_cv_mean_f1": summary.get("holdout_cv_mean_f1"),
            "holdout_cv_std_f1": summary.get("holdout_cv_std_f1"),
            **_macro_metrics(oof),
        })
    return records


def _collect_ae_classifier(runs_dir, pipeline, dataset, best_cfg):
    records = []
    if not runs_dir.exists():
        return records
    for details_path in sorted(runs_dir.glob("*/cv_run_details.json")):
        details = _load_json(details_path)
        if details is None:
            continue
        args = details.get("arguments", {})
        nc = args.get("num_classes")
        if nc not in NUM_CLASSES_RANGE:
            continue
        # Exclude outcome-sweep runs (those have a non-default score_name).
        if args.get("score_name", "QL2") != "QL2":
            continue
        if best_cfg and not _configs_match(args, best_cfg, _AE_MATCH_KEYS):
            continue
        run_dir = details_path.parent
        cv = details.get("cross_validation_metrics", {})
        bal = cv.get("oof_balanced_acc", cv.get("average_balanced_acc"))
        model = args.get("model_name")
        if not model and args.get("ae_weights"):
            model = Path(args["ae_weights"]).stem.split("_")[0]
        acc_pct = cv.get("average_val_acc")
        oof = details.get("oof_results", {})
        records.append({
            "pipeline": pipeline,
            "dataset": dataset,
            "run_name": run_dir.name,
            "model": model,
            "num_classes": nc,
            "balanced_accuracy": bal,
            "accuracy": None if acc_pct is None else acc_pct / 100.0,
            "val_loss": cv.get("average_val_loss"),
            "config": {k: args.get(k) for k in ("ae_weights", "lr", "dropout",
                                                  "weight_decay", "epochs", "folds",
                                                  "num_classes", "quantile")},
            "path": str(run_dir),
            **_macro_metrics(oof),
        })
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_leaderboard(write=True):
    best = _best_configs_from_leaderboard()

    records = []

    # End-to-end (2D + 3D): discover every dataset subdir under runs/ so any
    # dataset (brainwear, brainwear_png, brats_png, brats, ...) is picked up
    # automatically. The main leaderboard labels all end-to-end runs
    # 'end_to_end' regardless of 2D/3D, so we look the best config up under that
    # label but emit the disambiguated pipeline name ('end_to_end_2d' for PNG).
    if END_TO_END_RUNS.exists():
        for ds_dir in sorted(p.name for p in END_TO_END_RUNS.iterdir() if p.is_dir()):
            pipeline = "end_to_end_2d" if ds_dir.endswith("_png") else "end_to_end"
            records += _collect_end_to_end(
                ds_dir, pipeline, best.get(("end_to_end", ds_dir)))

    # Autoencoder-classifier: the 3D (brainwear) and 2D (brainwear_png) trees are
    # separate and the run dir does not encode the dataset, so map each tree to
    # its dataset explicitly. Add a brats_png mapping here once such runs exist.
    records += _collect_ae_classifier(
        AE_CLASSIFIER_RUNS, "ae_classifier", "brainwear",
        best.get(("ae_classifier", "brainwear")))
    records += _collect_ae_classifier(
        AE_CLASSIFIER_2D_RUNS, "ae_classifier_2d", "brainwear_png",
        best.get(("ae_classifier_2d", "brainwear_png")))

    records.sort(key=lambda r: (r["pipeline"], r["dataset"], r["num_classes"] or 99))
    best_record = next(
        (r for r in sorted(records, key=lambda r: -(r["balanced_accuracy"] or 0))
         if r["balanced_accuracy"] is not None),
        None,
    )

    leaderboard = {
        "generated": date.today().isoformat(),
        "primary_metric": PRIMARY_METRIC,
        "caveat": CAVEAT,
        "n_runs": len(records),
        "best": best_record,
        "ranked": records,
    }
    if write:
        with open(LEADERBOARD_PATH, "w") as f:
            json.dump(leaderboard, f, indent=2)
        print("Num-classes leaderboard written to {} ({} runs)".format(LEADERBOARD_PATH, len(records)))
    return leaderboard


def print_table(leaderboard):
    ranked = leaderboard["ranked"]
    if not ranked:
        print("No num-classes sweep runs found.")
        return
    hdr = ("{:>3}  {:<14} {:<12} {:<9} {:>3} {:>8} {:>7} {:>7} {:>9}  run_name"
           .format("#", "pipeline", "dataset", "model", "cls", "bal_acc", "f1", "acc", "val_loss"))
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(ranked, 1):
        bal = "{:.4f}".format(r["balanced_accuracy"]) if r["balanced_accuracy"] is not None else "  --  "
        f1  = "{:.4f}".format(r["f1"])               if r["f1"]               is not None else "  -- "
        acc = "{:.4f}".format(r["accuracy"])          if r["accuracy"]          is not None else "  -- "
        vl  = "{:.4f}".format(r["val_loss"])          if r["val_loss"]          is not None else "   --  "
        print("{:>3}  {:<14} {:<12} {:<9} {:>3} {:>8} {:>7} {:>7} {:>9}  {}".format(
            i, r["pipeline"], str(r["dataset"]), str(r["model"]),
            str(r["num_classes"]), bal, f1, acc, vl, r["run_name"]))
    if leaderboard["best"]:
        b = leaderboard["best"]
        print("\nBest: {}/{} nc={} -> balanced_accuracy={:.4f}".format(
            b["pipeline"], b["dataset"], b["num_classes"], b["balanced_accuracy"]))
    print("\nNote: {}".format(leaderboard["caveat"]))


if __name__ == "__main__":
    print_table(build_leaderboard(write=True))
