"""Offline generator for `src/autoencodix/tuning/data/initial_configs.json`.

Dev-only script -- requires `syne-tune` (see the `dev` dependency group in
`pyproject.toml`). It is never imported by the shipped `autoencodix` package.

It fits Syne Tune's `ZeroShotTransfer` scheduler against the BBOmix benchmark
blackbox data (105k autoencodix training runs) to produce a small ranked
portfolio of hyperparameter configs per (architecture, dataset-or-"combined",
objective[, budget]), and serializes the result to a static JSON artifact
shipped with the package.

Generalization note: nothing here is specific to TCGA/SCHC. Each entry in
`BLACKBOX_MANIFEST` below just names one blackbox and which metric plays the
"downstream" role (an aggregate performance score, maximized, final epoch
only) versus the "reconstruction" role (minimized, per-epoch/multi-fidelity)
for that blackbox. Adding coverage for a brand new dataset means adding a
manifest entry pointing at that dataset's own blackbox and its own metric
names -- no other code in this script needs to change.

Usage:
    python benchmarking/tuning/generate_initial_configs.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from syne_tune.blackbox_repository.blackbox_tabular import (
    BlackboxTabular,
    deserialize as deserialize_tabular,
)
from syne_tune.blackbox_repository.repository import repository_path
from syne_tune.optimizer.schedulers.transfer_learning.transfer_learning_task_evaluation import (
    TransferLearningTaskEvaluations,
)
from syne_tune.config_space import Domain
from syne_tune.optimizer.schedulers.transfer_learning.zero_shot import ZeroShotTransfer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("generate_initial_configs")


def _patched_sample_random_config(self, config_space: Dict[str, Any]) -> Dict[str, Any]:
    """Workaround for a bug in syne-tune==0.16.0's ZeroShotTransfer: it calls
    `self.random_state` inside `_create_surrogate_transfer_learning_evaluations`
    (triggered by `use_surrogates=True`), which runs *before* `self.random_state`
    is ever assigned in `__init__`, raising AttributeError. This lazily creates
    and caches the RandomState instead of relying on __init__ having set it.
    Dev-only patch, applied only in this generation script -- never shipped."""
    rng = getattr(self, "random_state", None)
    if rng is None:
        rng = np.random.RandomState(getattr(self, "random_seed", None))
        self.random_state = rng
    return {
        k: v.sample(random_state=rng) if isinstance(v, Domain) else v
        for k, v in config_space.items()
    }


ZeroShotTransfer._sample_random_config = _patched_sample_random_config

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "src" / "autoencodix" / "tuning" / "data" / "initial_configs.json"
README_PATH = Path(__file__).resolve().parent / "README.md"

BUDGET_GRID = [10, 25, 50, 100, 150, 200, 300]
TOP_K_STORED = 10  # ranked configs kept per portfolio leaf

# Fixed values held constant during the BBOmix sweep (see the BBOmix archive's
# `autoencodix_package_bbomix/benchmarking/configs/search_space.yaml`), merged
# into every proposed config.
FIXED_HPS: Dict[str, Any] = {
    "epochs": 300,
    "checkpoint_interval": 300,
    "loss_reduction": "sum",
}

# HP fields the blackbox config space stores as Float (or that a surrogate
# candidate may sample as float) but the Config schema requires as int.
INT_FIELDS = ["n_layers", "batch_size", "latent_dim", "k_filter"]

# --- manifest ---------------------------------------------------------------
# One entry per (architecture, dataset) blackbox usable as transfer-learning
# source data. This is the only place dataset-specific knowledge lives -- the
# rest of the script is generic over whatever is listed here.
BLACKBOX_MANIFEST: List[Dict[str, Any]] = [
    {
        "blackbox_name": f"bbomix_{arch}_{dataset}",
        "architecture": arch,
        "dataset": dataset,
        "downstream_metric": "metric_avg_ml_task_performance",
        "reconstruction_metric": "metric_valid_recon_loss",
    }
    for arch in ("vanillix", "varix", "ontix", "disentanglix")
    for dataset in ("tcga", "schc")
]

_blackbox_cache: Dict[str, Dict[str, BlackboxTabular]] = {}


def _load_bbomix_blackbox(name: str) -> Dict[str, BlackboxTabular]:
    """Load a locally-cached bbomix_* blackbox directly, bypassing Syne Tune's
    registered-blackbox allowlist (bbomix blackboxes are BBOmix-specific, not
    part of Syne Tune's own registry)."""
    if name not in _blackbox_cache:
        _blackbox_cache[name] = deserialize_tabular(repository_path / name)
    return _blackbox_cache[name]


def _sanitize_hyperparameters(hp_df: pd.DataFrame) -> pd.DataFrame:
    out = hp_df.copy()
    for col in out.columns:
        if pd.api.types.is_integer_dtype(out[col]):
            out[col] = out[col].astype(int)
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].astype(float)
        else:
            out[col] = out[col].astype(str)
    return out


def _task_evaluations(
    bb: BlackboxTabular, metric: str, epoch_idx: Optional[int]
) -> Optional[TransferLearningTaskEvaluations]:
    """Build a TransferLearningTaskEvaluations for one blackbox task, sliced at
    `epoch_idx` (0-indexed) or the final epoch when `epoch_idx is None`."""
    try:
        metric_index = bb.objectives_names.index(metric)
    except ValueError:
        return None

    evals = bb.objectives_evaluations  # (H, S, E, O)
    single = evals[..., metric_index : metric_index + 1]
    if epoch_idx is None:
        sliced = single[:, :, -1:, :]
    else:
        if epoch_idx >= single.shape[2]:
            return None
        sliced = single[:, :, epoch_idx : epoch_idx + 1, :]

    if np.all(np.isnan(sliced)):
        return None

    return TransferLearningTaskEvaluations(
        hyperparameters=_sanitize_hyperparameters(bb.hyperparameters),
        configuration_space=bb.configuration_space,
        objectives_evaluations=sliced,
        objectives_names=[metric],
    )


def _build_transfer_evaluations(
    entries: List[Dict[str, Any]],
    metric_field: str,
    epoch_idx: Optional[int] = None,
    exclude_task_key: Optional[str] = None,
) -> Dict[str, TransferLearningTaskEvaluations]:
    tle: Dict[str, TransferLearningTaskEvaluations] = {}
    for entry in entries:
        bb_dict = _load_bbomix_blackbox(entry["blackbox_name"])
        metric = entry[metric_field]
        for task_name, bb in bb_dict.items():
            task_key = f"{entry['blackbox_name']}/{task_name}"
            if task_key == exclude_task_key:
                continue
            task_eval = _task_evaluations(bb, metric, epoch_idx)
            if task_eval is not None:
                tle[task_key] = task_eval
    return tle


def _fit_and_rank(
    config_space: Dict[str, Any],
    metric: str,
    do_minimize: bool,
    tle: Dict[str, TransferLearningTaskEvaluations],
    num_configs: int = TOP_K_STORED,
) -> List[Dict[str, Any]]:
    if not tle:
        return []
    scheduler = ZeroShotTransfer(
        config_space=config_space,
        metric=metric,
        do_minimize=do_minimize,
        transfer_learning_evaluations=tle,
        use_surrogates=True,
    )
    configs: List[Dict[str, Any]] = []
    for _ in range(num_configs):
        cfg = scheduler.get_config()
        if cfg is None:
            break
        configs.append(dict(cfg))
    return configs


def _postprocess(hp: Dict[str, Any], epochs: int) -> Dict[str, Any]:
    out = dict(hp)
    for field in INT_FIELDS:
        if field in out and out[field] is not None:
            out[field] = int(round(out[field]))
    out.update(FIXED_HPS)
    if epochs < FIXED_HPS["epochs"]:
        out["epochs"] = epochs
        out["checkpoint_interval"] = epochs
    return out


def _nearest_neighbor_percentile(
    bb: BlackboxTabular,
    metric: str,
    epoch_idx: Optional[int],
    do_minimize: bool,
    hp: Dict[str, Any],
) -> Optional[float]:
    """Approximate how good a recommended config is on a held-out task: find the
    nearest real evaluated config (by normalized Euclidean distance in HP
    space) and report what percentile of all real configs it beats.

    Informational only -- this is a nearest-neighbor proxy, not an exact
    surrogate evaluation of the recommended config."""
    try:
        metric_index = bb.objectives_names.index(metric)
    except ValueError:
        return None

    evals = bb.objectives_evaluations[..., metric_index]  # (H, S, E)
    idx = evals.shape[2] - 1 if epoch_idx is None else epoch_idx
    if idx >= evals.shape[2]:
        return None
    scores = np.nanmean(evals[:, :, idx], axis=1)  # (H,), averaged over seeds
    valid = ~np.isnan(scores)
    if valid.sum() == 0:
        return None

    hp_df = bb.hyperparameters
    columns = [c for c in hp_df.columns if c in hp]
    if not columns:
        return None
    ranges = hp_df[columns].max() - hp_df[columns].min()
    ranges = ranges.replace(0, 1.0)
    norm_df = (hp_df[columns] - hp_df[columns].min()) / ranges
    target = np.array([(hp[c] - hp_df[c].min()) / ranges[c] for c in columns])
    dists = np.linalg.norm(norm_df.values - target, axis=1)
    dists = np.where(valid, dists, np.inf)
    nn_idx = int(np.argmin(dists))
    nn_score = scores[nn_idx]

    valid_scores = scores[valid]
    if do_minimize:
        percentile = float((valid_scores >= nn_score).mean() * 100)
    else:
        percentile = float((valid_scores <= nn_score).mean() * 100)
    return percentile


def _architectures() -> List[str]:
    seen = []
    for entry in BLACKBOX_MANIFEST:
        if entry["architecture"] not in seen:
            seen.append(entry["architecture"])
    return seen


def _datasets_for(arch: str) -> List[str]:
    return [e["dataset"] for e in BLACKBOX_MANIFEST if e["architecture"] == arch]


def run_validation(arch: str, report_lines: List[str]) -> None:
    """Leave-one-task-out validation: for each held-out task, refit on the rest
    and record the top-1 recommendation's nearest-neighbor percentile on the
    held-out task. Informational only -- not asserted in CI."""
    entries = [e for e in BLACKBOX_MANIFEST if e["architecture"] == arch]
    config_space = _load_bbomix_blackbox(entries[0]["blackbox_name"])[
        list(_load_bbomix_blackbox(entries[0]["blackbox_name"]))[0]
    ].configuration_space

    objective_specs = [("downstream", "downstream_metric", None, False)]
    for budget in BUDGET_GRID:
        objective_specs.append(
            (f"reconstruction@{budget}", "reconstruction_metric", budget - 1, True)
        )

    for label, metric_field, epoch_idx, do_minimize in objective_specs:
        percentiles: List[float] = []
        for entry in entries:
            bb_dict = _load_bbomix_blackbox(entry["blackbox_name"])
            metric = entry[metric_field]
            for task_name, bb in bb_dict.items():
                task_key = f"{entry['blackbox_name']}/{task_name}"
                tle = _build_transfer_evaluations(
                    entries, metric_field, epoch_idx, exclude_task_key=task_key
                )
                top = _fit_and_rank(
                    config_space, metric, do_minimize, tle, num_configs=1
                )
                if not top:
                    continue
                pct = _nearest_neighbor_percentile(
                    bb, metric, epoch_idx, do_minimize, top[0]
                )
                if pct is not None:
                    percentiles.append(pct)
        if percentiles:
            msg = (
                f"  {label}: mean percentile {np.mean(percentiles):.1f} "
                f"(n={len(percentiles)} held-out tasks)"
            )
        else:
            msg = f"  {label}: no held-out evaluations available"
        logger.info(msg)
        report_lines.append(msg)


def build_portfolios_for_architecture(arch: str) -> Dict[str, Any]:
    entries = [e for e in BLACKBOX_MANIFEST if e["architecture"] == arch]
    first_bb = _load_bbomix_blackbox(entries[0]["blackbox_name"])
    config_space = first_bb[list(first_bb)[0]].configuration_space

    result: Dict[str, Any] = {}

    def _downstream_portfolio(scoped_entries):
        tle = _build_transfer_evaluations(scoped_entries, "downstream_metric")
        raw = _fit_and_rank(config_space, "metric_avg_ml_task_performance", False, tle)
        return [_postprocess(hp, epochs=FIXED_HPS["epochs"]) for hp in raw]

    def _reconstruction_portfolio(scoped_entries):
        out = {}
        for budget in BUDGET_GRID:
            tle = _build_transfer_evaluations(
                scoped_entries, "reconstruction_metric", epoch_idx=budget - 1
            )
            raw = _fit_and_rank(config_space, "metric_valid_recon_loss", True, tle)
            out[str(budget)] = [_postprocess(hp, epochs=budget) for hp in raw]
        return out

    result["combined"] = {
        "downstream": _downstream_portfolio(entries),
        "reconstruction": _reconstruction_portfolio(entries),
    }
    for dataset in _datasets_for(arch):
        scoped = [e for e in entries if e["dataset"] == dataset]
        result[dataset] = {
            "downstream": _downstream_portfolio(scoped),
            "reconstruction": _reconstruction_portfolio(scoped),
        }
    return result


def main() -> None:
    report_lines = ["# Initial-config generation report", ""]
    artifact: Dict[str, Any] = {}

    for arch in _architectures():
        logger.info("=== %s ===", arch)
        report_lines.append(f"## {arch}")
        run_validation(arch, report_lines)
        artifact[arch] = build_portfolios_for_architecture(arch)
        report_lines.append("")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(artifact, indent=2))
    logger.info("Wrote %s", OUTPUT_PATH)

    README_PATH.write_text("\n".join(report_lines) + "\n")
    logger.info("Wrote %s", README_PATH)


if __name__ == "__main__":
    main()
