"""Matched 5-fold CV for the BrainWear 2D baseline pipelines.

Analogous to ``aacbr/eval_brats_2d_cv.py`` but for the ResNet baselines
instead of AA-CBR.  Provides:

  configs_from_leaderboard()  -- pick best 2D baseline config per outcome score
  run_configs()               -- retrain those configs for n_bins ∈ {2,3,4,5}
  log_results()               -- append results to brainwear_baseline_cv.json

Two pipelines are supported, matching the existing leaderboard entries:

  end_to_end_2d    -- torchvision ResNet-50 (5-channel PNG input), CE loss
  ae_classifier_2d -- pre-trained AE encoder (frozen) + trainable linear head

Both pipelines use StratifiedKFold(n_folds, shuffle=True, random_state=seed)
on the BrainWearPNGDataset, so the CV splits are reproducible and comparable
across n_bins values.

Usage example (run from FYP root):
    import baseline.brainwear_2d_baseline_cv as bcv
    configs = bcv.configs_from_leaderboard('baseline/brainwear_leaderboard.json')
    sweep   = bcv.run_configs(configs, n_bins_range=(2, 3, 4, 5))
    bcv.log_results(sweep, 'baseline/brainwear_baseline_cv.json')
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

FYP_ROOT = Path(__file__).resolve().parents[1]
if str(FYP_ROOT) not in sys.path:
    sys.path.insert(0, str(FYP_ROOT))

DATA_DIR   = str(FYP_ROOT.parent / "Processed_Brainwear_PNG")
SCORE_FILE = str(FYP_ROOT.parent / "eortc_scores.csv")

EVAL_SCRIPT = "baseline_2d_cv"

# Pipelines to consider when choosing best config per score.
_2D_PIPELINES = ["end_to_end_2d", "ae_classifier_2d"]

# Default AE encoder weights (used when ae_classifier_2d config lacks ae_weights).
_DEFAULT_AE_WEIGHTS = str(
    FYP_ROOT / "baseline" / "autoencoder" / "models" / "encoder_2d"
    / "resnet18_150e_0.0004lr.pt"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path) -> list | dict:
    return json.loads(Path(path).read_text())


def _load_eortc_csv(score_file: str, score_name: str) -> dict[str, float]:
    """Parse eortc_scores.csv and return {patient_id: float_score} for one score."""
    import csv as _csv
    result: dict[str, float] = {}
    with open(score_file, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        q_col = reader.fieldnames[0]
        for row in reader:
            if row[q_col].strip() != score_name:
                continue
            for pid, val in row.items():
                if pid == q_col:
                    continue
                pid = pid.strip()
                try:
                    result[pid] = float(val.strip())
                except (ValueError, TypeError):
                    pass
            break
    return result


class _RelabeledDataset(torch.utils.data.Dataset):
    """Wraps an image dataset and replaces its labels with a provided array."""
    def __init__(self, base, labels: np.ndarray):
        self._base   = base
        self._labels = labels

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        x, _ = self._base[idx]
        return x, int(self._labels[idx])


def _build_dataset(score_name: str, n_bins: int) -> tuple:
    """Load images and bin outcomes using np.quantile + np.digitize.

    Uses the same binning as the Slot+AA-CBR evaluation notebook so that
    class labels are directly comparable between the two pipelines.

    Returns (dataset, labels_array) where dataset yields (image, class_int).
    """
    from datasets.brainwear_png import BrainWearPNGDataset

    # Image-only dataset — gives us the canonical patient list with PNGs.
    img_ds = BrainWearPNGDataset(DATA_DIR, score_file=None)
    all_pids = img_ds.patient_folders

    # Raw outcome scores for this score_name.
    raw_scores = _load_eortc_csv(SCORE_FILE, score_name)

    # Keep only patients present in both image directory and score CSV.
    valid_mask = [pid in raw_scores for pid in all_pids]
    valid_idx  = [i for i, ok in enumerate(valid_mask) if ok]
    valid_pids = [all_pids[i] for i in valid_idx]

    raw = np.array([raw_scores[pid] for pid in valid_pids], dtype=float)

    # Identical binning to get_outcomes() in eval_trained_model_brainwear_2D.ipynb.
    thresholds = np.quantile(raw, np.linspace(0, 1, n_bins + 1)[1:-1]) if n_bins > 1 else np.array([])
    labels = np.digitize(raw, thresholds).astype(int)

    dist = np.bincount(labels, minlength=n_bins).tolist()
    print(f"  {score_name} k={n_bins}: n={len(valid_pids)}, dist={dist}, "
          f"thresholds={np.round(thresholds, 1).tolist()}")

    return _RelabeledDataset(Subset(img_ds, valid_idx), labels), labels


def _make_resnet2d(model_name: str, num_classes: int, in_channels: int = 5) -> nn.Module:
    """Torchvision ResNet with first conv adapted for in_channels."""
    import torchvision.models as tvm
    base = tvm.resnet18(weights=None) if model_name == "resnet18" else tvm.resnet50(weights=None)
    base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    base.fc = nn.Linear(base.fc.in_features, num_classes)
    return base


def _load_ae_classifier_2d(ae_weights: str, num_classes: int, dropout: float) -> nn.Module:
    """Return a ResNetClassifier2D with the encoder loaded from ae_weights."""
    from baseline.autoencoder.models_2d import ResNetAutoencoder2D, ResNetClassifier2D
    ae = ResNetAutoencoder2D(model_name="resnet18", in_channels=5)
    state = torch.load(ae_weights, map_location="cpu")
    # ae_weights may be encoder-only or full autoencoder; try both
    try:
        ae.encoder.load_state_dict(state)
    except RuntimeError:
        ae.load_state_dict(state, strict=False)
    return ResNetClassifier2D(ae, num_classes=num_classes, dropout=dropout,
                              unfreeze_encoder=False)


def _train_fold(
    model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
    epochs: int, lr: float, weight_decay: float, patience: int, device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Train model for one CV fold; return (y_true, y_pred) on the val set."""
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    no_improve = 0
    best_state: dict | None = None

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            criterion(model(x), y.long()).backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                val_loss += criterion(model(x.to(device)), y.to(device).long()).item()
        val_loss /= max(1, len(val_loader))

        sched.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            preds = model(x.to(device)).argmax(dim=1)
            y_true.extend(y.tolist())
            y_pred.extend(preds.cpu().tolist())

    return np.array(y_true), np.array(y_pred)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configs_from_leaderboard(
    lb_path,
    dataset: str = "brainwear_png",
    pipelines: list[str] | None = None,
    scores: list[str] | None = None,
) -> list[dict]:
    """Return the best 2D baseline config per outcome score.

    Filters ``lb_path`` (brainwear_leaderboard.json) to 2D entries matching
    *dataset* and *pipelines*, then picks the highest-F1 entry per score_name.
    Returns a list of config dicts ready to pass to ``run_configs()``.
    """
    if pipelines is None:
        pipelines = _2D_PIPELINES
    lb = _load_json(lb_path)
    rows = lb.get("ranked", lb) if isinstance(lb, dict) else lb
    rows = [
        r for r in rows
        if r.get("dataset") == dataset
        and r.get("pipeline") in pipelines
        and r.get("f1") is not None
    ]
    if scores:
        rows = [r for r in rows if r.get("score_name") in scores]

    best: dict[str, dict] = {}
    for r in rows:
        s = r.get("score_name")
        if s and (s not in best or r["f1"] > best[s]["f1"]):
            best[s] = r

    out = []
    for s, r in sorted(best.items()):
        cfg = dict(r.get("config", {}))
        cfg["pipeline"]   = r["pipeline"]
        cfg["score_name"] = s
        cfg["model"]      = r.get("model", cfg.get("model", "resnet50"))
        cfg["_k3_f1"]     = r["f1"]
        # Fill in defaults if missing; force quantile=True for the k-sweep so
        # all bins have comparable class frequencies regardless of original config.
        cfg.setdefault("lr", 1e-5)
        cfg.setdefault("weight_decay", 1e-4)
        cfg.setdefault("epochs", 30)
        cfg.setdefault("patience", 15)
        cfg["quantile"] = True   # override — equal-width bins produce degenerate distributions
        cfg.setdefault("dropout", 0.6)
        cfg.setdefault("ae_weights", _DEFAULT_AE_WEIGHTS)
        out.append(cfg)

    print(f"Best 2D baseline config per outcome score ({len(out)} scores):")
    for c in out:
        print(f"  {c['score_name']:6s}  pipeline={c['pipeline']:16s}  "
              f"model={c['model']:9s}  k=3 F1={c['_k3_f1']:.4f}")
    return out


def configs_from_hyper_leaderboard(
    hyper_lb_path,
    scores: list[str],
    dataset: str = "brainwear_png",
    pipelines: list[str] | None = None,
) -> list[dict]:
    """Return configs built from the *hyperparameter* sweep leaderboard.

    Unlike ``configs_from_leaderboard`` (which reads per-score outcome-sweep
    entries from brainwear_leaderboard.json), this function reads the general
    hyperparameter sweep entries (score_name=None) from ``leaderboard.json``
    and picks the highest-F1 pipeline+config across those runs.  The same best
    config is then reused for every requested outcome score.

    This is the correct source to use when the outcome-sweep config may have
    been set to a sub-optimal value during the manual sweep setup.
    """
    if pipelines is None:
        pipelines = _2D_PIPELINES
    lb = _load_json(hyper_lb_path)
    rows = lb.get("ranked", lb) if isinstance(lb, dict) else lb

    # General runs have score_name=None; they represent the hyper sweep.
    general = [
        r for r in rows
        if r.get("dataset") == dataset
        and r.get("pipeline") in pipelines
        and r.get("score_name") is None
        and r.get("f1") is not None
    ]
    if not general:
        raise ValueError(
            f"No score_name=None entries found for dataset={dataset!r} "
            f"pipelines={pipelines} in {hyper_lb_path}"
        )

    best_row = max(general, key=lambda r: r["f1"])
    print(
        f"Best general 2D config from hyperparameter sweep:\n"
        f"  pipeline={best_row['pipeline']}  model={best_row['model']}  "
        f"F1={best_row['f1']:.4f}\n"
        f"  hyperparams: {best_row['config']}"
    )

    out = []
    for s in scores:
        cfg = dict(best_row.get("config", {}))
        cfg["pipeline"]   = best_row["pipeline"]
        cfg["score_name"] = s
        cfg["model"]      = best_row.get("model", cfg.get("model", "resnet50"))
        cfg["_hyper_f1"]  = best_row["f1"]
        cfg["quantile"]   = True
        cfg.setdefault("patience", 15)
        cfg.setdefault("ae_weights", _DEFAULT_AE_WEIGHTS)
        out.append(cfg)

    print(f"\nBuilt {len(out)} configs (one per score) using these hyperparameters:")
    for c in out:
        print(f"  {c['score_name']:6s}  pipeline={c['pipeline']:16s}  "
              f"model={c['model']:9s}  lr={c['lr']}  dropout={c.get('dropout')}")
    return out


def _already_logged(lb_path, pipeline: str, score_name: str, n_bins: int,
                    quantile: bool = True, lr: float | None = None,
                    dropout: float | None = None) -> bool:
    """True if a matching entry exists in the target leaderboard.

    Checks pipeline, score_name, n_bins, and quantile; also checks lr and
    dropout when provided so that re-running with new hyperparameters is not
    skipped due to a stale entry with different values.
    """
    p = Path(lb_path)
    if not p.exists():
        return False
    records = _load_json(p)
    for r in records:
        c = r.get("config", {})
        if not (r.get("eval_script") == EVAL_SCRIPT
                and c.get("pipeline") == pipeline
                and c.get("score_name") == score_name
                and c.get("n_bins") == n_bins
                and bool(c.get("quantile", True)) == quantile):
            continue
        if lr is not None and c.get("lr") != lr:
            continue
        if dropout is not None and c.get("dropout") != dropout:
            continue
        return True
    return False


def run_configs(
    configs: list[dict],
    n_bins_range: tuple[int, ...] = (2, 3, 4, 5),
    n_folds: int = 5,
    seed: int = 0,
    lb_path=None,
    device: torch.device | None = None,
) -> list[dict]:
    """Run 5-fold CV for each (config, n_bins) pair.

    If *lb_path* is given, entries already present there are skipped.
    Returns a list of result dicts (one per completed run).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = []
    total = len(configs) * len(n_bins_range)
    done = 0

    for cfg in configs:
        pipeline   = cfg["pipeline"]
        score_name = cfg["score_name"]
        model_name = cfg.get("model", "resnet50")
        lr         = float(cfg.get("lr", 1e-5))
        wd         = float(cfg.get("weight_decay", 1e-4))
        epochs     = int(cfg.get("epochs", 30))
        patience   = int(cfg.get("patience", 15))
        dropout    = float(cfg.get("dropout", 0.6))
        ae_weights = cfg.get("ae_weights", _DEFAULT_AE_WEIGHTS)

        for n_bins in n_bins_range:
            done += 1
            tag = f"[{done}/{total}] {pipeline}/{score_name}/k={n_bins}"

            if lb_path and _already_logged(lb_path, pipeline, score_name, n_bins,
                                           quantile=True, lr=lr,
                                           dropout=dropout if pipeline == "ae_classifier_2d" else None):
                print(f"{tag}  → already logged, skipping")
                continue

            print(f"\n{tag}  training...")
            t0 = time.time()

            dataset, labels = _build_dataset(score_name, n_bins)
            n_patients = len(dataset)

            dist = np.bincount(labels, minlength=n_bins).tolist()
            if min(dist) < n_folds:
                print(f"  Skip: class distribution {dist} too small for {n_folds}-fold CV")
                continue

            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            fold_f1s, fold_accs, fold_bals = [], [], []
            fold_precs, fold_recs = [], []

            for fold_idx, (tr_idx, val_idx) in enumerate(
                    tqdm(kf.split(np.arange(n_patients), labels),
                         total=n_folds, desc=f"  Folds"), start=1):
                tr_loader = DataLoader(Subset(dataset, tr_idx.tolist()),
                                       batch_size=8, shuffle=True,
                                       num_workers=2, pin_memory=True)
                val_loader = DataLoader(Subset(dataset, val_idx.tolist()),
                                        batch_size=8, shuffle=False,
                                        num_workers=2, pin_memory=True)

                if pipeline == "end_to_end_2d":
                    model = _make_resnet2d(model_name, num_classes=n_bins).to(device)
                elif pipeline == "ae_classifier_2d":
                    model = _load_ae_classifier_2d(ae_weights, n_bins, dropout).to(device)
                else:
                    raise ValueError(f"Unknown pipeline: {pipeline!r}")

                y_true, y_pred = _train_fold(
                    model, tr_loader, val_loader, epochs, lr, wd, patience, device)

                p, r, f, _ = precision_recall_fscore_support(
                    y_true, y_pred, average="macro", zero_division=0)
                fold_f1s.append(float(f))
                fold_accs.append(float(accuracy_score(y_true, y_pred)))
                fold_bals.append(float(balanced_accuracy_score(y_true, y_pred)))
                fold_precs.append(float(p))
                fold_recs.append(float(r))

            elapsed = time.time() - t0
            result = {
                "eval_script": EVAL_SCRIPT,
                "config": {
                    "pipeline":    pipeline,
                    "score_name":  score_name,
                    "n_bins":      n_bins,
                    "model":       model_name,
                    "lr":          lr,
                    "weight_decay": wd,
                    "epochs":      epochs,
                    "patience":    patience,
                    "quantile":    True,
                    "dropout":     dropout if pipeline == "ae_classifier_2d" else None,
                    "ae_weights":  ae_weights if pipeline == "ae_classifier_2d" else None,
                    "n_folds":     n_folds,
                    "seed":        seed,
                },
                "data_stats": {
                    "n_patients":        n_patients,
                    "class_distribution": dist,
                },
                "metrics": {
                    "cv_mean_f1":              float(np.mean(fold_f1s)),
                    "cv_std_f1":               float(np.std(fold_f1s)),
                    "cv_mean_accuracy":        float(np.mean(fold_accs)),
                    "cv_std_accuracy":         float(np.std(fold_accs)),
                    "cv_mean_balanced_accuracy": float(np.mean(fold_bals)),
                    "cv_std_balanced_accuracy":  float(np.std(fold_bals)),
                    "cv_mean_precision":       float(np.mean(fold_precs)),
                    "cv_std_precision":        float(np.std(fold_precs)),
                    "cv_mean_recall":          float(np.mean(fold_recs)),
                    "cv_std_recall":           float(np.std(fold_recs)),
                    "fold_f1s":                fold_f1s,
                    "fold_accs":               fold_accs,
                    "fold_balanced_accs":      fold_bals,
                },
                "elapsed_minutes": round(elapsed / 60, 2),
                "notes": "",
            }
            results.append(result)
            print(f"  → cv_f1={result['metrics']['cv_mean_f1']:.4f}"
                  f"±{result['metrics']['cv_std_f1']:.4f}"
                  f"  ({elapsed/60:.1f} min)")

    return results


def log_results(sweep: list[dict], lb_path) -> None:
    """Append sweep results to *lb_path* (brainwear_baseline_cv.json)."""
    from aacbr.results_logger import log_result
    for entry in sweep:
        log_result(entry, leaderboard_path=lb_path)
    print(f"\nLogged {len(sweep)} results to {lb_path}")
