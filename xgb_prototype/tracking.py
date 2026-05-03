"""
tracking.py — Post-training bookkeeping for the XGBoost prototype.

Covers four concerns that are neither evaluation nor plotting:

  _MLflowRun           : context-manager wrapping an MLflow run (U25).
  write_requirements_lock : capture installed package versions (U26).
  register_model       : append a run to the JSON model registry (N6).
  train_summary        : human-readable log table + JSON run report (U20).
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .data import DriftReport   # avoids circular import at runtime

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# UPGRADE 25 — MLflow experiment tracking
# ─────────────────────────────────────────────

class _MLflowRun:
    """U25: Context manager wrapping an MLflow run. No-ops gracefully if mlflow absent or URI unset."""

    def __init__(self, run_id: str, timestamp: str, mlflow_uri: str | None, experiment: str) -> None:
        self._active = False
        self._run    = None
        if mlflow_uri is None:
            return
        try:
            import mlflow as _mlflow
            self._mlflow = _mlflow
            _mlflow.set_tracking_uri(mlflow_uri)
            _mlflow.set_experiment(experiment)
            self._run = _mlflow.start_run(run_name=f"{timestamp}_{run_id}")
            self._active = True
            log.info("  MLflow run started: %s (experiment=%s)",
                     self._run.info.run_id, experiment)
        except ImportError:
            log.warning("  mlflow not installed — experiment tracking disabled.")
        except Exception as e:
            log.warning("  MLflow setup failed (skipping): %s", e)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._active:
            try:
                self._mlflow.end_run()
            except Exception:
                pass

    def log_params(self, params: dict) -> None:
        if not self._active:
            return
        try:
            self._mlflow.log_params(params)
        except Exception as e:
            log.debug("MLflow log_params failed: %s", e)

    def log_metrics(self, metrics: dict) -> None:
        if not self._active:
            return
        try:
            self._mlflow.log_metrics(metrics)
        except Exception as e:
            log.debug("MLflow log_metrics failed: %s", e)

    def log_artifact(self, path: "Path | str") -> None:
        if not self._active:
            return
        try:
            self._mlflow.log_artifact(str(path))
        except Exception as e:
            log.debug("MLflow log_artifact failed: %s", e)

    def log_config(self, cfg: dict) -> None:
        if not self._active:
            return
        try:
            flat = {k: str(v) for k, v in cfg.items()}
            self._mlflow.log_params(flat)
        except Exception as e:
            log.debug("MLflow log_config failed: %s", e)


# ─────────────────────────────────────────────
# UPGRADE 26 — Reproducibility lock file
# ─────────────────────────────────────────────

def write_requirements_lock(run_id: str, output_dir: Path) -> Path:
    """U26: Capture all installed package versions to models/requirements_<run_id>.txt."""
    pkgs = sorted(
        f"{dist.name}=={dist.version}"
        for dist in importlib.metadata.distributions()
    )
    content = "\n".join(pkgs) + "\n"
    path = output_dir / f"requirements_{run_id}.txt"
    path.write_text(content)
    log.info("  Requirements lock saved → %s (%d packages)", path, len(pkgs))
    return path


# ─────────────────────────────────────────────
# N6 — JSON model registry
# ─────────────────────────────────────────────

def _load_registry(registry_file: Path) -> list[dict]:
    """Load registry from disk; return empty list if absent or corrupt."""
    try:
        if registry_file.exists():
            return json.loads(registry_file.read_text())
    except Exception:
        pass
    return []


def _save_registry(records: list[dict], registry_file: Path) -> None:
    registry_file.write_text(json.dumps(records, indent=2))


def _registry_status(metrics: dict, task: str) -> str:
    """Decide production_ready / needs_review from eval metrics.

    Rules (conservative defaults):
      classification: ROC-AUC ≥ 0.80  AND  AUPRC ≥ 0.50  → production_ready
      regression:     R² ≥ 0.70                            → production_ready
      Otherwise → needs_review
    """
    if task == "regression":
        return "production_ready" if metrics.get("r2", 0) >= 0.70 else "needs_review"
    auc   = metrics.get("roc_auc", 0)
    auprc = metrics.get("auprc",   0)
    if auc >= 0.80 and auprc >= 0.50:
        return "production_ready"
    return "needs_review"


def register_model(
    run_id: str,
    timestamp: str,
    model_path: Path,
    eval_metrics: dict,
    best_params: dict,
    best_n: int,
    best_threshold: float,
    task: str,
    config_snapshot: dict,
    output_dir: Path,
    baseline_results: list[dict] | None = None,
    ensemble_summary: dict | None = None,
    threshold_policy: dict | None = None,
) -> None:
    """N6: Append this run to the persistent JSON model registry.

    Each entry contains:
      run_id, timestamp, artifact path, eval metrics,
      best hyperparams, n_estimators, threshold,
      status (production_ready | needs_review),
      task, and a copy of the config snapshot.

    The registry file is <output_dir>/model_registry.json.
    All previous runs are preserved; newest entry is last.
    """
    registry_file = output_dir / "model_registry.json"
    records = _load_registry(registry_file)
    status  = _registry_status(eval_metrics, task)

    entry = {
        "run_id":            f"{timestamp}_{run_id}",
        "timestamp":         timestamp,
        "artifact":          str(model_path),
        "task":              task,
        "status":            status,
        "eval_metrics":      {k: round(float(v), 6) for k, v in eval_metrics.items()},
        "best_params":       best_params,
        "best_n_estimators": best_n,
        "best_threshold":    round(float(best_threshold), 6),
        "threshold_policy":  threshold_policy or {},
        "baseline_results":  baseline_results or [],
        "ensemble_summary":  ensemble_summary or {},
        "config":            config_snapshot,
    }
    records.append(entry)
    _save_registry(records, registry_file)

    emoji = "✓" if status == "production_ready" else "⚠"
    log.info(
        "  [N6] Registry updated — %s %s  (%d total runs)  → %s",
        emoji, status, len(records), registry_file,
    )


# ─────────────────────────────────────────────
# UPGRADE 20 — Run summary helper
# ─────────────────────────────────────────────

def train_summary(
    run_id: str,
    timestamp: str,
    model_path: Path,
    eval_metrics: dict,
    best_params: dict,
    best_n: int,
    best_threshold: float,
    drift_report: "DriftReport",
    output_dir: Path,
    plot_output_dir: Path,
    task: str,
    target_col: str,
    baseline_results: list[dict] | None = None,
    ensemble_summary: dict | None = None,
    threshold_policy: dict | None = None,
    feature_schema: dict | None = None,
    artifact_paths: dict | None = None,
) -> None:
    """
    UPGRADE 20: Concise post-run summary table + JSON run report.

    The JSON report is machine-readable so CI/CD pipelines can parse it
    and fail a build when key metrics regress below a threshold.
    """
    sep = "─" * 60
    log.info("\n%s", sep)
    log.info("  RUN SUMMARY")
    log.info("%s", sep)
    log.info("  Run ID          : %s_%s", timestamp, run_id)
    log.info("  Artifact        : %s", model_path)
    log.info("  Best n_estimators: %d", best_n)
    log.info("  Best threshold  : %.4f", best_threshold)
    for k, v in eval_metrics.items():
        log.info("  %-16s: %.4f", k, v)
    if drift_report.any_drift:
        log.warning(
            "  Drift detected  : num=%s  cat=%s",
            drift_report.drifted_numerical, drift_report.drifted_categorical,
        )
    else:
        log.info("  Drift           : none detected")
    log.info("  Plots saved to  : %s", plot_output_dir)
    log.info("%s", sep)

    report = {
        "run_id":            f"{timestamp}_{run_id}",
        "artifact":          str(model_path),
        "eval_metrics":      eval_metrics,
        "baseline_results":  baseline_results or [],
        "ensemble_summary":  ensemble_summary or {},
        "best_params":       best_params,
        "best_n_estimators": best_n,
        "best_threshold":    best_threshold,
        "threshold_policy":  threshold_policy or {},
        "feature_schema":    feature_schema or {},
        "drift":             drift_report.to_dict(),
        "task":              task,
        "target_col":        target_col,
        "artifact_paths":    artifact_paths or {},
    }
    report_path = output_dir / f"run_{timestamp}_{run_id}.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info("  Run report saved → %s", report_path)