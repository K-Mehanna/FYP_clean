"""Aggregate outcome-score sweep results into brainwear_leaderboard.json.

Scans all four BrainWear pipelines (3D/2D end-to-end and 3D/2D AE-classifier)
for runs that include a ``score_name`` field in their saved arguments.  Old
hyperparameter-sweep runs (which pre-date ``score_name``) are excluded so the
leaderboard is a clean comparison of EORTC outcome scales under fixed, optimal
hyperparameters.

    python -m baseline.sweeps.brainwear_aggregate
"""

import json
import sys
from datetime import date
from pathlib import Path

FYP_ROOT = Path(__file__).resolve().parents[2]
if str(FYP_ROOT) not in sys.path:
    sys.path.insert(0, str(FYP_ROOT))

BASELINE_DIR = FYP_ROOT / "baseline"
END_TO_END_RUNS = BASELINE_DIR / "end-to-end" / "runs"
AE_CLASSIFIER_RUNS = BASELINE_DIR / "autoencoder" / "models" / "classifier"
AE_CLASSIFIER_2D_RUNS = BASELINE_DIR / "autoencoder" / "models" / "classifier_2d"
LEADERBOARD_PATH = BASELINE_DIR / "brainwear_leaderboard.json"

PRIMARY_METRIC = "balanced_accuracy"
CAVEAT = (
    "balanced_accuracy is only directly comparable within the same num_classes and pipeline; "
    "score_name indicates the EORTC scale used as the prediction target."
)


def _load_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [warn] could not read {path}: {exc}")
        return None


def _macro_metrics(oof: dict) -> dict:
    """Extract macro-averaged precision, recall, F1 from an oof_results dict.

    New runs have these as top-level keys; old runs carry them only inside
    classification_report["macro avg"], so we fall back there.
    """
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


def _collect_end_to_end(dataset_dir_name: str, pipeline: str) -> list[dict]:
    """Collect end-to-end runs for brainwear or brainwear_png."""
    records = []
    runs_dir = END_TO_END_RUNS / dataset_dir_name
    if not runs_dir.exists():
        return records
    for results_path in sorted(runs_dir.glob("*/results.json")):
        results = _load_json(results_path)
        if results is None:
            continue
        run_dir = results_path.parent
        args = _load_json(run_dir / "args.json") or {}
        if "score_name" not in args:
            continue  # pre-outcome-sweep run; skip
        summary = results.get("summary", {})
        oof = results.get("oof_results", {})
        records.append(
            {
                "pipeline": pipeline,
                "dataset": dataset_dir_name,
                "score_name": args["score_name"],
                "run_name": run_dir.name,
                "model": summary.get("model") or args.get("model"),
                "num_classes": args.get("num_bins"),
                "balanced_accuracy": oof.get("balanced_accuracy"),
                "accuracy": summary.get("accuracy_mean"),
                **_macro_metrics(oof),
                "val_loss": summary.get("best_val_loss_mean"),
                "config": {
                    k: args.get(k)
                    for k in ("model", "lr", "weight_decay", "epochs", "n_splits",
                              "num_bins", "quantile", "strategy", "score_name")
                },
                "path": str(run_dir),
            }
        )
    return records


def _collect_ae_classifier(runs_dir: Path, pipeline: str, dataset: str) -> list[dict]:
    """Collect AE-classifier runs (3D or 2D)."""
    records = []
    if not runs_dir.exists():
        return records
    for details_path in sorted(runs_dir.glob("*/cv_run_details.json")):
        details = _load_json(details_path)
        if details is None:
            continue
        args = details.get("arguments", {})
        if "score_name" not in args:
            continue  # pre-outcome-sweep run; skip
        run_dir = details_path.parent
        cv = details.get("cross_validation_metrics", {})
        bal = cv.get("oof_balanced_acc", cv.get("average_balanced_acc"))
        model = args.get("model_name")
        if not model and args.get("ae_weights"):
            model = Path(args["ae_weights"]).stem.split("_")[0]
        acc_pct = cv.get("average_val_acc")
        oof = details.get("oof_results", {})
        records.append(
            {
                "pipeline": pipeline,
                "dataset": dataset,
                "score_name": args["score_name"],
                "run_name": run_dir.name,
                "model": model,
                "num_classes": args.get("num_classes"),
                "balanced_accuracy": bal,
                "accuracy": None if acc_pct is None else acc_pct / 100.0,
                **_macro_metrics(oof),
                "val_loss": cv.get("average_val_loss"),
                "config": {
                    k: args.get(k)
                    for k in ("ae_weights", "lr", "dropout", "weight_decay", "epochs",
                              "folds", "num_classes", "quantile", "score_name")
                },
                "path": str(run_dir),
            }
        )
    return records


def build_leaderboard(write: bool = True) -> dict:
    """Collect all outcome-sweep runs, rank by balanced accuracy, write JSON."""
    records = (
        _collect_end_to_end("brainwear", "end_to_end")
        + _collect_end_to_end("brainwear_png", "end_to_end_2d")
        + _collect_ae_classifier(AE_CLASSIFIER_RUNS, "ae_classifier", "brainwear")
        + _collect_ae_classifier(AE_CLASSIFIER_2D_RUNS, "ae_classifier_2d", "brainwear_png")
    )

    def sort_key(r):
        bal = r["balanced_accuracy"]
        return (0, -bal) if bal is not None else (1, 0.0)

    records.sort(key=sort_key)
    best = next((r for r in records if r["balanced_accuracy"] is not None), None)

    leaderboard = {
        "generated": date.today().isoformat(),
        "primary_metric": PRIMARY_METRIC,
        "caveat": CAVEAT,
        "n_runs": len(records),
        "best": best,
        "ranked": records,
    }
    if write:
        with open(LEADERBOARD_PATH, "w") as f:
            json.dump(leaderboard, f, indent=2)
        print(f"Brainwear leaderboard written to {LEADERBOARD_PATH} ({len(records)} runs)")
    return leaderboard


def print_table(leaderboard: dict) -> None:
    ranked = leaderboard["ranked"]
    if not ranked:
        print("No outcome-sweep runs found.")
        return
    hdr = (
        f"{'#':>3}  {'pipeline':<14} {'score':<6} {'model':<9} "
        f"{'cls':>3} {'bal_acc':>8} {'acc':>7} {'val_loss':>9}  run_name"
    )
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(ranked, 1):
        bal = f"{r['balanced_accuracy']:.4f}" if r["balanced_accuracy"] is not None else "  --  "
        acc = f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "  -- "
        vl = f"{r['val_loss']:.4f}" if r["val_loss"] is not None else "   --  "
        print(
            f"{i:>3}  {r['pipeline']:<14} {r['score_name']:<6} {str(r['model']):<9} "
            f"{str(r['num_classes']):>3} {bal:>8} {acc:>7} {vl:>9}  {r['run_name']}"
        )
    if leaderboard["best"]:
        b = leaderboard["best"]
        print(
            f"\nBest: {b['pipeline']}/{b['score_name']}/{b['run_name']} "
            f"-> balanced_accuracy={b['balanced_accuracy']:.4f}"
        )
    print(f"\nNote: {leaderboard['caveat']}")


if __name__ == "__main__":
    print_table(build_leaderboard(write=True))
