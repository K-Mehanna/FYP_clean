"""Hyperparameter-sweep toolkit for the baseline models.

All training/aggregation logic lives in the ``.py`` modules here; the notebooks
(``run_sweep.ipynb``, ``aggregate.ipynb``) and batch scripts (``run_sweep.slurm``,
``run_sweep.pbs``) are thin wrappers that import and call these modules so there
is a single source of truth.
"""
