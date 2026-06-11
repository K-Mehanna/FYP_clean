"""Sequential hyperparameter-sweep driver for the baseline models.

Reads a small JSON grid config, expands the cartesian product of the grid into
full configs (merged onto a shared ``base``), and runs each one **in-process,
sequentially**, then refreshes the leaderboard.

Config schema (see baseline/sweeps/configs/*.json):
    {
      "sweep":   "end_to_end" | "ae_classifier",
      "dataset": "brainwear" | "brats",        # end_to_end only
      "base":    { ...fixed params... },
      "grid":    { "param": [v1, v2, ...], ... }
    }

    python -m baseline.sweeps.run_sweep --config baseline/sweeps/configs/sweep_brainwear.json
"""

import argparse
import itertools
import json
import sys
import traceback
from pathlib import Path

FYP_ROOT = Path(__file__).resolve().parents[2]
if str(FYP_ROOT) not in sys.path:
    sys.path.insert(0, str(FYP_ROOT))

from baseline.autoencoder import train_autoencoder_2d
from baseline.sweeps import aggregate, brainwear_aggregate, train_ae_classifier, train_ae_classifier_2d, train_end_to_end, train_end_to_end_2d


def expand_grid(base: dict, grid: dict) -> list[dict]:
    """Cartesian product of grid lists, each merged onto base."""
    if not grid:
        return [dict(base)]
    keys = list(grid.keys())
    combos = itertools.product(*(grid[k] for k in keys))
    return [{**base, **dict(zip(keys, combo))} for combo in combos]


def _result_exists(sweep: str, cfg: dict) -> Path | None:
    """Return the existing result file for this config, if already run."""
    if sweep in ("end_to_end", "end_to_end_2d"):
        module = train_end_to_end if sweep == "end_to_end" else train_end_to_end_2d
        f = module.run_dir_for(cfg) / "results.json"
    elif sweep == "autoencoder_2d":
        f = train_autoencoder_2d.result_path_for(cfg)
    elif sweep == "ae_classifier_2d":
        f = train_ae_classifier_2d.run_dir_for(cfg) / "cv_run_details.json"
    else:
        f = train_ae_classifier.run_dir_for(cfg) / "cv_run_details.json"
    return f if f.exists() else None


def _prepare_cfg(sweep: str, dataset: str | None, raw: dict) -> dict:
    """Fill defaults so run_dir naming / training have every key they need."""
    if sweep == "end_to_end":
        ds = dataset or raw.get("dataset", "brainwear")
        defaults = vars(train_end_to_end.build_parser().parse_args(["--dataset", ds]))
        merged = {**defaults, **raw, "dataset": ds}
        return train_end_to_end.cfg_from_args(argparse.Namespace(**merged))
    if sweep == "end_to_end_2d":
        ds = dataset or raw.get("dataset", "brainwear_png")
        defaults = vars(train_end_to_end_2d.build_parser().parse_args(["--dataset", ds]))
        merged = {**defaults, **raw, "dataset": ds}
        return train_end_to_end_2d.cfg_from_args(argparse.Namespace(**merged))
    if sweep == "autoencoder_2d":
        defaults = vars(train_autoencoder_2d.build_parser().parse_args([]))
        merged = {**defaults, **raw}
        return merged
    if sweep == "ae_classifier_2d":
        defaults = vars(train_ae_classifier_2d.build_parser().parse_args(["--ae_weights", raw["ae_weights"]]))
        merged = {**defaults, **raw}
        return train_ae_classifier_2d.cfg_from_args(argparse.Namespace(**merged))
    # ae_classifier (3D)
    defaults = vars(train_ae_classifier.build_parser().parse_args(["--ae_weights", raw["ae_weights"]]))
    merged = {**defaults, **raw}
    return train_ae_classifier.cfg_from_args(argparse.Namespace(**merged))


def run(config_path: str, force: bool = False, refresh_leaderboard: bool = True) -> None:
    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = FYP_ROOT / config_path
    spec = json.loads(cfg_file.read_text())
    sweep = spec["sweep"]
    dataset = spec.get("dataset")
    configs = expand_grid(spec.get("base", {}), spec.get("grid", {}))

    print(f"=== Sweep '{sweep}' ({dataset or 'brainwear'}): {len(configs)} configs from {cfg_file.name} ===")
    if sweep == "end_to_end":
        runner = train_end_to_end.run_cv
    elif sweep == "end_to_end_2d":
        runner = train_end_to_end_2d.run_cv
    elif sweep == "autoencoder_2d":
        runner = train_autoencoder_2d.run_pretraining
    elif sweep == "ae_classifier_2d":
        runner = train_ae_classifier_2d.run_cv
    else:
        runner = train_ae_classifier.run_cv

    statuses = []
    for i, raw in enumerate(configs, 1):
        cfg = _prepare_cfg(sweep, dataset, raw)
        existing = _result_exists(sweep, cfg)
        tag = f"[{i}/{len(configs)}]"
        if existing and not force:
            print(f"{tag} SKIP (already done): {existing.parent.name}")
            statuses.append((cfg, "skipped"))
            continue
        print(f"\n{tag} RUN: {raw}")
        try:
            runner(cfg)
            statuses.append((cfg, "ok"))
        except Exception:  # keep the sweep alive if one config fails
            print(f"{tag} FAILED:\n{traceback.format_exc()}")
            statuses.append((cfg, "failed"))

    n_ok      = sum(1 for _, s in statuses if s == "ok")
    n_skipped = sum(1 for _, s in statuses if s == "skipped")
    n_failed  = sum(1 for _, s in statuses if s == "failed")
    print(f"\n=== Sweep summary: {n_ok} succeeded | {n_skipped} skipped | {n_failed} failed ===")
    for cfg, status in statuses:
        print(f"  {status:<8} {cfg.get('dataset', 'brainwear')} {cfg.get('model') or cfg.get('model_name', '')} "
              f"lr={cfg.get('lr')} wd={cfg.get('weight_decay')} dropout={cfg.get('dropout', '-')}")

    if refresh_leaderboard:
        aggregate.print_table(aggregate.build_leaderboard(write=True))
        brainwear_aggregate.print_table(brainwear_aggregate.build_leaderboard(write=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sequential baseline sweep driver.")
    p.add_argument("--config", required=True, help="Path to a sweep JSON config (abs or relative to FYP/).")
    p.add_argument("--force", action="store_true", help="Re-run configs even if results already exist.")
    p.add_argument("--no-leaderboard", action="store_true", help="Skip refreshing the leaderboard at the end.")
    a = p.parse_args()
    run(a.config, force=a.force, refresh_leaderboard=not a.no_leaderboard)
