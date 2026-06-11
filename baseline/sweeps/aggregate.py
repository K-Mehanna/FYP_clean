"""Aggregate every baseline run into one ranked leaderboard.

Scans both pipelines' per-run metric files, normalises their differing schemas
into a common record, ranks by **balanced accuracy** (the user-chosen metric),
and writes ``baseline/leaderboard.json``. Picks up runs produced by the
notebooks as well as by the sweep trainers.

    python -m baseline.sweeps.aggregate
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
LEADERBOARD_PATH = BASELINE_DIR / "leaderboard.json"

PRIMARY_METRIC = "f1"
CAVEAT = (
    "f1 (macro-averaged) is only directly comparable within the same num_classes; "
    "each record carries num_classes so you can compare like-for-like."
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


def _collect_end_to_end() -> list[dict]:
    records = []
    for results_path in sorted(END_TO_END_RUNS.glob("*/*/results.json")):
        results = _load_json(results_path)
        if results is None:
            continue
        run_dir = results_path.parent
        args = _load_json(run_dir / "args.json") or {}
        summary = results.get("summary", {})
        oof = results.get("oof_results", {})
        records.append(
            {
                "pipeline": "end_to_end",
                "dataset": run_dir.parent.name,  # runs/<dataset>/<run_name>
                "run_name": run_dir.name,
                "score_name": args.get("score_name"),
                "model": summary.get("model") or args.get("model"),
                "num_classes": args.get("num_bins"),
                "balanced_accuracy": oof.get("balanced_accuracy"),
                "accuracy": summary.get("accuracy_mean"),
                **_macro_metrics(oof),
                "val_loss": summary.get("best_val_loss_mean"),
                "config": {k: args.get(k) for k in ("model", "lr", "weight_decay", "epochs", "n_splits", "num_bins", "quantile", "strategy")},
                "path": str(run_dir),
                "needs_rerun": oof.get("balanced_accuracy") is None,
            }
        )
    return records


def _collect_ae_classifier() -> list[dict]:
    records = []
    for details_path in sorted(AE_CLASSIFIER_RUNS.glob("*/cv_run_details.json")):
        details = _load_json(details_path)
        if details is None:
            continue
        run_dir = details_path.parent
        args = details.get("arguments", {})
        cv = details.get("cross_validation_metrics", {})
        bal = cv.get("oof_balanced_acc", cv.get("average_balanced_acc"))  # new keys only
        model = args.get("model_name")
        if not model and args.get("ae_weights"):
            model = Path(args["ae_weights"]).stem.split("_")[0]
        acc_pct = cv.get("average_val_acc")  # stored as a percentage
        oof = details.get("oof_results", {})
        records.append(
            {
                "pipeline": "ae_classifier",
                "dataset": "brainwear",
                "run_name": run_dir.name,
                "score_name": args.get("score_name"),
                "model": model,
                "num_classes": args.get("num_classes"),
                "balanced_accuracy": bal,
                "accuracy": None if acc_pct is None else acc_pct / 100.0,
                **_macro_metrics(oof),
                "val_loss": cv.get("average_val_loss"),
                "config": {k: args.get(k) for k in ("ae_weights", "lr", "dropout", "weight_decay", "epochs", "folds", "num_classes", "quantile")},
                "path": str(run_dir),
                # Old notebook runs predate balanced accuracy -> flag for a rerun.
                "needs_rerun": bal is None,
            }
        )
    return records


def _collect_ae_classifier_2d() -> list[dict]:
    records = []
    for details_path in sorted(AE_CLASSIFIER_2D_RUNS.glob("*/cv_run_details.json")):
        details = _load_json(details_path)
        if details is None:
            continue
        run_dir = details_path.parent
        args = details.get("arguments", {})
        cv = details.get("cross_validation_metrics", {})
        bal = cv.get("oof_balanced_acc", cv.get("average_balanced_acc"))
        model = args.get("model_name")
        if not model and args.get("ae_weights"):
            model = Path(args["ae_weights"]).stem.split("_")[0]
        acc_pct = cv.get("average_val_acc")
        oof = details.get("oof_results", {})
        records.append(
            {
                "pipeline": "ae_classifier_2d",
                "dataset": "brainwear_png",
                "run_name": run_dir.name,
                "score_name": args.get("score_name"),
                "model": model,
                "num_classes": args.get("num_classes"),
                "balanced_accuracy": bal,
                "accuracy": None if acc_pct is None else acc_pct / 100.0,
                **_macro_metrics(oof),
                "val_loss": cv.get("average_val_loss"),
                "config": {k: args.get(k) for k in ("ae_weights", "lr", "dropout", "weight_decay", "epochs", "folds", "num_classes", "quantile")},
                "path": str(run_dir),
                "needs_rerun": bal is None,
            }
        )
    return records


def build_leaderboard(write: bool = True) -> dict:
    """Collect all runs, rank by balanced accuracy, optionally write the JSON."""
    records = _collect_end_to_end() + _collect_ae_classifier() + _collect_ae_classifier_2d()

    # Sort: runs with an f1 score first (desc), then flagged runs.
    def sort_key(r):
        f1 = r["f1"]
        return (0, -f1) if f1 is not None else (1, 0.0)

    records.sort(key=sort_key)
    ranked = records
    best = next((r for r in ranked if r["f1"] is not None), None)

    leaderboard = {
        "generated": date.today().isoformat(),
        "primary_metric": PRIMARY_METRIC,
        "caveat": CAVEAT,
        "n_runs": len(ranked),
        "best": best,
        "ranked": ranked,
    }
    if write:
        with open(LEADERBOARD_PATH, "w") as f:
            json.dump(leaderboard, f, indent=2)
        print(f"Leaderboard written to {LEADERBOARD_PATH}")
    return leaderboard


def print_table(leaderboard: dict) -> None:
    ranked = leaderboard["ranked"]
    if not ranked:
        print("No runs found.")
        return
    hdr = f"{'#':>2}  {'pipeline':<13} {'dataset':<9} {'model':<9} {'cls':>3} {'f1':>7} {'bal_acc':>8} {'acc':>7} {'val_loss':>9}  run_name"
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(ranked, 1):
        f1 = f"{r['f1']:.4f}" if r["f1"] is not None else "  -- "
        bal = f"{r['balanced_accuracy']:.4f}" if r["balanced_accuracy"] is not None else "  --  "
        acc = f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "  -- "
        vl = f"{r['val_loss']:.4f}" if r["val_loss"] is not None else "   --  "
        flag = "  (needs_rerun)" if r["needs_rerun"] else ""
        print(f"{i:>2}  {r['pipeline']:<13} {str(r['dataset']):<9} {str(r['model']):<9} {str(r['num_classes']):>3} {f1:>7} {bal:>8} {acc:>7} {vl:>9}  {r['run_name']}{flag}")
    if leaderboard["best"]:
        b = leaderboard["best"]
        print(f"\nBest by {leaderboard['primary_metric']}: {b['pipeline']}/{b['dataset']}/{b['run_name']} "
              f"-> f1={b['f1']:.4f} ({b['num_classes']} classes)")
    print(f"\nNote: {leaderboard['caveat']}")


if __name__ == "__main__":
    print_table(build_leaderboard(write=True))
