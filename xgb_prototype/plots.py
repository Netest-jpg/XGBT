"""plots.py — All Plotly and Matplotlib visualisation functions."""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from ._render import heatmap_large, heatmap_large_html, scatter_large, scatter_large_html
from .settings import (
    OUTLIER_CONTAMINATION, PCA_VARIANCE, PDP_TOP_N,
    PLOT_OUTPUT_DIR, RANDOM_STATE, RASTER_FORMAT, TARGET_COL,
)

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*scattermapbox.*", category=DeprecationWarning)


def _remove_stale_plot_variants(stem: str, keep) -> None:
    keep_path = PLOT_OUTPUT_DIR / keep if isinstance(keep, str) else keep
    for suffix in (".html", ".png", ".jpg", ".jpeg", ".webp"):
        path = PLOT_OUTPUT_DIR / f"{stem}{suffix}"
        if path != keep_path and path.exists():
            try:
                path.unlink()
            except OSError:
                pass


def plot_pr_curve(y_test: np.ndarray, y_proba: np.ndarray, threshold: float) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    auprc    = average_precision_score(y_test, y_proba)
    baseline = float(np.mean(y_test))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines",
                             name=f"PR curve (AUPRC={auprc:.4f})",
                             fill="tozeroy", fillcolor="rgba(68,119,170,0.12)",
                             line=dict(color="#4477AA", width=2)))
    fig.add_hline(y=baseline, line_dash="dash", line_color="gray",
                  annotation_text=f"Baseline ({baseline:.4f})")
    if len(thresholds) > 0:
        idx = np.argmin(np.abs(thresholds - threshold))
        fig.add_trace(go.Scatter(x=[recall[idx]], y=[precision[idx]], mode="markers",
                                 marker=dict(color="#CC3333", size=10),
                                 name=f"Threshold={threshold:.4f}"))
    fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall",
                      yaxis_title="Precision", xaxis=dict(range=[0, 1]),
                      yaxis=dict(range=[0, 1.05]), template="simple_white")
    path = PLOT_OUTPUT_DIR / "pr_curve.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    log.info("  PR curve → %s", path)


def plot_roc_curve(y_test: np.ndarray, y_proba: np.ndarray) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score   = roc_auc_score(y_test, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"ROC (AUC={auc_score:.4f})",
                             fill="tozeroy", fillcolor="rgba(68,170,119,0.10)",
                             line=dict(color="#44AA77", width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color="gray", dash="dash"), name="Random"))
    fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR",
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02]),
                      template="simple_white")
    path = PLOT_OUTPUT_DIR / "roc_curve.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    log.info("  ROC curve → %s", path)


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    from sklearn.metrics import confusion_matrix
    cm        = confusion_matrix(y_test, y_pred)
    labels    = sorted(np.unique(np.concatenate([y_test, y_pred])).tolist())
    labels_s  = [str(l) for l in labels]
    n         = len(labels)
    coords    = list(range(n))
    cm_norm   = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    # Use numeric coordinates so axis labels move with the plot during zoom/pan.
    # tickvals/ticktext map those integers back to the class names.
    fig = go.Figure(go.Heatmap(z=cm_norm, x=coords, y=coords,
                               colorscale="Blues", showscale=True, text=cm,
                               texttemplate="%{text}",
                               hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{text}<extra></extra>"))
    axis_common = dict(
        tickvals=coords, ticktext=labels_s,
        fixedrange=False,
    )
    fig.update_layout(title="Confusion Matrix (row-normalised colour, raw count labels)",
                      xaxis_title="Predicted", yaxis_title="True",
                      template="simple_white",
                      xaxis=dict(**axis_common),
                      yaxis=dict(**axis_common, autorange="reversed"))
    path = PLOT_OUTPUT_DIR / "confusion_matrix.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    log.info("  Confusion matrix → %s", path)


def plot_residuals(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    from plotly.subplots import make_subplots
    from sklearn.metrics import r2_score
    residuals = y_test - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    r2   = float(r2_score(y_test, y_pred))
    mn   = float(min(y_test.min(), y_pred.min()))
    mx   = float(max(y_test.max(), y_pred.max()))
    fig  = make_subplots(rows=1, cols=2, subplot_titles=["Predicted vs Actual", "Residual Distribution"])
    fig.add_trace(go.Scatter(x=y_pred, y=y_test, mode="markers",
                             marker=dict(color="#4477AA", opacity=0.4, size=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                             line=dict(color="red", dash="dash"), name="Perfect fit"), row=1, col=1)
    fig.add_trace(go.Histogram(x=residuals, nbinsx=60, marker_color="#4477AA", opacity=0.7), row=1, col=2)
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=2)
    fig.update_layout(title=f"Residual Diagnostics — RMSE={rmse:.4f}, R²={r2:.4f}",
                      template="simple_white", showlegend=False)
    path = PLOT_OUTPUT_DIR / "residuals.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    log.info("  Residuals → %s", path)


def plot_optuna_diagnostics(study) -> None:
    try:
        from optuna.visualization import plot_optimization_history, plot_param_importances
        for fig, fname in [
            (plot_optimization_history(study), "optuna_history.html"),
            (plot_param_importances(study),    "optuna_importance.html"),
        ]:
            fig.update_layout(template="simple_white")
            path = PLOT_OUTPUT_DIR / fname
            fig.write_html(str(path), include_plotlyjs="cdn")
            log.info("  Optuna %s → %s", fname, path)
    except Exception as e:
        log.warning("  Optuna visualisation failed (skipping): %s", e)


def plot_feature_importance(
    pipeline: Pipeline,
    num_cols: list[str],
    ohe_cat_cols: list[str],
    te_cat_cols: list[str],
    use_pca: bool,
) -> None:
    log.info("[8/9] Plotting feature importance (XGBoost gain)...")
    model    = pipeline.named_steps["model"]
    prep     = pipeline.named_steps["preprocessor"]
    num_pipe = prep.named_transformers_.get("num")

    if use_pca and num_pipe is not None:
        pca    = num_pipe.named_steps["pca"]
        n_comp = pca.n_components_ if hasattr(pca, "n_components_") else pca.components_.shape[0]
        feat_names: list[str] = [f"PC{i+1}" for i in range(n_comp)]
    else:
        feat_names = list(num_cols)
    if ohe_cat_cols:
        ohe = prep.named_transformers_["cat"].named_steps["encoder"]
        feat_names.extend(ohe.get_feature_names_out(ohe_cat_cols).tolist())
    if te_cat_cols:
        feat_names.extend(te_cat_cols)

    base_model = model
    if hasattr(model, "estimator"):          base_model = model.estimator
    elif hasattr(model, "calibrated_classifiers_"): base_model = model.calibrated_classifiers_[0].estimator

    scores = base_model.feature_importances_
    n_feat = min(len(feat_names), len(scores))
    imp_df = (pd.DataFrame({"feature": feat_names[:n_feat], "importance": scores[:n_feat]})
              .sort_values("importance", ascending=False).head(20))
    sorted_df = imp_df.sort_values("importance", ascending=True).reset_index(drop=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_h = max(4.8, 0.32 * len(sorted_df) + 1.2)
    fig, ax = plt.subplots(figsize=(8.8, fig_h))
    ax.barh(sorted_df["feature"], sorted_df["importance"], color="#4477AA", alpha=0.88)
    ax.set_title("Top 20 Feature Importances - XGBoost gain")
    ax.set_xlabel("Gain importance")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    max_imp = float(sorted_df["importance"].max()) if len(sorted_df) else 0.0
    if max_imp > 0:
        ax.set_xlim(0, max_imp * 1.12)
        for y_pos, val in enumerate(sorted_df["importance"]):
            ax.text(val + max_imp * 0.015, y_pos, f"{val:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    suffix = ".jpg" if RASTER_FORMAT == "jpeg" else f".{RASTER_FORMAT}"
    path = PLOT_OUTPUT_DIR / f"feature_importance{suffix}"
    save_kwargs = {"dpi": 140, "bbox_inches": "tight", "format": "jpeg" if RASTER_FORMAT == "jpeg" else RASTER_FORMAT}
    if RASTER_FORMAT == "jpeg":
        save_kwargs["facecolor"] = "white"
    fig.savefig(str(path), **save_kwargs)
    plt.close(fig)
    log.info("  Feature importance → %s", path)


def plot_pca_diagnostics(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, task: str,
) -> None:
    num_pipe = pipeline.named_steps["preprocessor"].named_transformers_.get("num")
    if num_pipe is None or "pca" not in num_pipe.named_steps:
        return

    pca    = num_pipe.named_steps["pca"]
    evr    = pca.explained_variance_ratio_
    cumv   = np.cumsum(evr)
    n_keep = len(evr)
    xs     = list(range(1, n_keep + 1))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=evr * 100, name="Individual", marker_color="#4477AA", opacity=0.75))
    fig.add_trace(go.Scatter(x=xs, y=cumv * 100, name="Cumulative", mode="lines+markers",
                              line=dict(color="#CC3333", width=2), marker=dict(size=5)))
    fig.add_hline(y=PCA_VARIANCE * 100, line_dash="dot", line_color="gray")
    fig.update_layout(title="PCA scree plot", xaxis_title="Component", yaxis_title="Variance (%)",
                      template="simple_white", yaxis=dict(range=[0, 105]))
    path = PLOT_OUTPUT_DIR / "pca_scree.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    log.info("  PCA scree → %s", path)

    if n_keep < 2:
        return

    preprocessor = pipeline.named_steps["preprocessor"]
    X_proc = preprocessor.transform(X_train)
    pc1, pc2 = X_proc[:, 0], X_proc[:, 1]

    path2 = scatter_large(
        pc1, pc2,
        color=y_train.values,
        title="PCA - first two principal components",
        x_label="PC1",
        y_label="PC2",
        output_name="pca_2d",
        color_label=TARGET_COL if task == "regression" else "Class",
        cmap="viridis",
    )
    log.info("  PCA 2-D → %s", path2)

    if n_keep >= 3:
        _MAX_3D = 10_000
        n_total = X_proc.shape[0]
        rng = np.random.default_rng(RANDOM_STATE)
        if n_total > _MAX_3D:
            if task == "classification":
                y_arr = y_train.values
                classes_u, counts = np.unique(y_arr, return_counts=True)
                keep_idx = np.concatenate([
                    rng.choice(np.where(y_arr == cls)[0],
                               size=max(1, int(_MAX_3D * cnt / n_total)), replace=False)
                    for cls, cnt in zip(classes_u, counts)
                ])
            else:
                keep_idx = rng.choice(n_total, size=_MAX_3D, replace=False)
            X_3d = X_proc[keep_idx, :3]; y_3d = y_train.values[keep_idx]
        else:
            X_3d = X_proc[:, :3]; y_3d = y_train.values

        _label = f"PCA — first 3 PCs (n={len(y_3d):,})"
        if task == "classification":
            fig3d = px.scatter_3d(x=X_3d[:, 0], y=X_3d[:, 1], z=X_3d[:, 2],
                                  color=y_3d.astype(str),
                                  labels={"x": "PC1", "y": "PC2", "z": "PC3", "color": "Class"},
                                  title=_label, opacity=0.65, template="simple_white")
        else:
            fig3d = px.scatter_3d(x=X_3d[:, 0], y=X_3d[:, 1], z=X_3d[:, 2],
                                  color=y_3d, color_continuous_scale="Viridis",
                                  labels={"x": "PC1", "y": "PC2", "z": "PC3", "color": TARGET_COL},
                                  title=_label, opacity=0.65, template="simple_white")
        fig3d.update_traces(marker=dict(size=2))
        path3d = PLOT_OUTPUT_DIR / "pca_3d.html"
        fig3d.write_html(str(path3d), include_plotlyjs="cdn")
        log.info("  PCA 3-D → %s", path3d)


def plot_permutation_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_cols: list[str],
    ohe_cat_cols: list[str],
    te_cat_cols: list[str],
    metric,
    task: str,
    X_test_proc: np.ndarray | None = None,
) -> None:
    n_rows = len(X_test_proc) if X_test_proc is not None else len(X_test)
    n_features = (X_test_proc.shape[1] if X_test_proc is not None else X_test.shape[1])
    n_repeats = 3 if n_rows * max(n_features, 1) > 100_000 else 5
    max_samples = min(1.0, 5_000 / max(n_rows, 1))
    log.info("  Computing permutation importance (n_repeats=%d, max_samples=%s)...",
             n_repeats, f"{max_samples:.3f}" if max_samples < 1.0 else "all")
    scoring_map = {"roc_auc": "roc_auc", "auprc": "average_precision",
                   "macro_f1": "f1_macro", "weighted_f1": "f1_weighted", "r2": "r2"}
    scoring = scoring_map.get(metric.name, "r2")

    if X_test_proc is not None:
        result = permutation_importance(
            pipeline.named_steps["model"], X_test_proc, y_test,
            scoring=scoring, n_repeats=n_repeats, random_state=RANDOM_STATE,
            n_jobs=-1, max_samples=max_samples,
        )
        prep = pipeline.named_steps["preprocessor"]
        all_names: list[str] = []
        if num_cols:
            num_pipe = prep.named_transformers_.get("num")
            if num_pipe and "pca" in num_pipe.named_steps:
                pca    = num_pipe.named_steps["pca"]
                n_comp = pca.n_components_ if hasattr(pca, "n_components_") else pca.components_.shape[0]
                all_names.extend([f"PC{i+1}" for i in range(n_comp)])
            else:
                all_names.extend(num_cols)
        if ohe_cat_cols:
            ohe = prep.named_transformers_["cat"].named_steps["encoder"]
            all_names.extend(ohe.get_feature_names_out(ohe_cat_cols).tolist())
        if te_cat_cols:
            all_names.extend(te_cat_cols)
        feat_names = all_names[:result.importances_mean.shape[0]]
    else:
        result = permutation_importance(
            pipeline, X_test, y_test,
            scoring=scoring, n_repeats=n_repeats, random_state=RANDOM_STATE,
            n_jobs=-1, max_samples=max_samples,
        )
        feat_names = list(X_test.columns)

    means = result.importances_mean
    stds  = result.importances_std
    n_feat = min(len(feat_names), len(means))
    imp_df = (pd.DataFrame({"feature": feat_names[:n_feat], "mean": means[:n_feat], "std": stds[:n_feat]})
              .sort_values("mean", ascending=True).tail(20).reset_index(drop=True))
    coords = list(range(len(imp_df)))
    # Numeric y-axis so tick labels travel with bars during zoom/pan.
    fig = go.Figure()
    fig.add_trace(go.Bar(x=imp_df["mean"], y=coords, orientation="h",
                         error_x=dict(type="data", array=imp_df["std"].tolist(), visible=True),
                         marker_color="#CC3333", opacity=0.8, name="Permutation importance"))
    fig.update_layout(title=f"Top 20 Permutation Importances (test set, scoring={scoring})",
                      xaxis_title=f"Mean decrease in {scoring}",
                      yaxis=dict(tickvals=coords, ticktext=imp_df["feature"].tolist(),
                                 title="", fixedrange=False),
                      template="simple_white")
    path = PLOT_OUTPUT_DIR / "permutation_importance.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    log.info("  Permutation importance → %s", path)


def plot_learning_curve(
    pipeline: Pipeline, X_trainval: pd.DataFrame,
    y_trainval: pd.Series, metric, task: str,
    X_trainval_proc: np.ndarray | None = None,
) -> None:
    log.info("  Computing fast learning curve (single holdout, native XGBoost)...")
    scoring_map = {"roc_auc": "roc_auc", "auprc": "average_precision",
                   "macro_f1": "f1_macro", "weighted_f1": "f1_weighted", "r2": "r2"}
    scoring = scoring_map.get(metric.name, "r2")
    train_sizes_frac = np.array([0.10, 0.18, 0.32, 0.56, 1.0])

    try:
        from .pipeline import _as_xgb_matrix, _score_booster, _xgb_model_params

        y_arr = np.asarray(y_trainval)
        if X_trainval_proc is None:
            preprocessor = pipeline.named_steps["preprocessor"]
            X_proc = _as_xgb_matrix(preprocessor.fit_transform(X_trainval, y_arr))
        else:
            X_proc = _as_xgb_matrix(X_trainval_proc)

        _LC_ROW_CAP = 20_000
        rng = np.random.default_rng(RANDOM_STATE)
        if len(X_proc) > _LC_ROW_CAP:
            if task == "classification":
                keep = []
                classes, counts = np.unique(y_arr, return_counts=True)
                for cls, cnt in zip(classes, counts):
                    cls_idx = np.where(y_arr == cls)[0]
                    n_keep = max(1, int(round(_LC_ROW_CAP * cnt / len(y_arr))))
                    keep.append(rng.choice(cls_idx, size=min(n_keep, len(cls_idx)), replace=False))
                keep_idx = np.concatenate(keep)
            else:
                keep_idx = rng.choice(len(X_proc), size=_LC_ROW_CAP, replace=False)
            X_proc, y_arr = X_proc[keep_idx], y_arr[keep_idx]

        if task == "classification":
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
            train_idx, val_idx = next(splitter.split(X_proc, y_arr))
        else:
            train_idx, val_idx = train_test_split(
                np.arange(len(X_proc)), test_size=0.25, random_state=RANDOM_STATE
            )
        X_tr_all, y_tr_all = X_proc[train_idx], y_arr[train_idx]
        X_val_lc, y_val_lc = X_proc[val_idx], y_arr[val_idx]

        model = pipeline.named_steps["model"]
        base_model = model.estimator if hasattr(model, "estimator") else model
        params = base_model.get_xgb_params()
        n_estimators = min(80, int(getattr(base_model, "n_estimators", 100) or 100))
        xgb_params = _xgb_model_params(
            task, metric, {**params, "nthread": 1},
            n_classes=int(len(np.unique(y_arr))) if task == "classification" else None,
        )
        dval = xgb.DMatrix(X_val_lc, label=y_val_lc)
        train_sizes, tr_mean, vl_mean = [], [], []
        for frac in train_sizes_frac:
            n_take = max(20, int(round(len(X_tr_all) * frac)))
            if task == "classification":
                sub = []
                classes, counts = np.unique(y_tr_all, return_counts=True)
                for cls, cnt in zip(classes, counts):
                    cls_idx = np.where(y_tr_all == cls)[0]
                    cls_take = max(1, int(round(n_take * cnt / len(y_tr_all))))
                    sub.append(rng.choice(cls_idx, size=min(cls_take, len(cls_idx)), replace=False))
                idx = np.concatenate(sub)
            else:
                idx = rng.choice(len(X_tr_all), size=min(n_take, len(X_tr_all)), replace=False)
            dtrain = xgb.DMatrix(X_tr_all[idx], label=y_tr_all[idx])
            booster = xgb.train(
                xgb_params, dtrain, num_boost_round=max(1, n_estimators),
                evals=[(dval, "validation")], verbose_eval=False,
            )
            train_sizes.append(len(idx))
            tr_mean.append(_score_booster(booster, dtrain, y_tr_all[idx], task, metric))
            vl_mean.append(_score_booster(booster, dval, y_val_lc, task, metric))
        train_sizes = np.asarray(train_sizes)
        tr_mean = np.asarray(tr_mean)
        vl_mean = np.asarray(vl_mean)
    except Exception as e:
        log.warning("  Learning curve failed (skipping): %s", e); return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes.tolist(), y=tr_mean.tolist(), mode="lines+markers",
                             name="Train", line=dict(color="#4477AA", width=2),
                             ))
    fig.add_trace(go.Scatter(x=train_sizes.tolist(), y=vl_mean.tolist(), mode="lines+markers",
                             name="Val (holdout)", line=dict(color="#CC3333", width=2)))
    fig.update_layout(title=f"Learning Curve — {scoring}",
                      xaxis_title="Training set size", yaxis_title=scoring,
                      template="simple_white")
    path = PLOT_OUTPUT_DIR / "learning_curve.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    log.info("  Learning curve → %s", path)


def plot_threshold_sweep(
    pipeline: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, metric,
    X_val_proc: np.ndarray | None = None,
) -> None:
    from sklearn.metrics import f1_score, precision_score, recall_score
    if not (metric.needs_proba and metric.name in ("auprc", "roc_auc")):
        return
    model_step = pipeline.named_steps["model"] if X_val_proc is not None else None
    y_proba = (model_step.predict_proba(X_val_proc)[:, 1]
               if model_step is not None else pipeline.predict_proba(X_val)[:, 1])
    y_arr = np.array(y_val)
    rows = []
    for t in np.linspace(0.01, 0.99, 100):
        y_pred = (y_proba >= t).astype(int)
        rows.append({"threshold": round(float(t), 3),
                     "precision": float(precision_score(y_arr, y_pred, zero_division=0)),
                     "recall":    float(recall_score(y_arr, y_pred, zero_division=0)),
                     "f1":        float(f1_score(y_arr, y_pred, zero_division=0)),
                     "support":   int(y_pred.sum())})
    sweep_df = pd.DataFrame(rows)
    metrics_to_plot = ["precision", "recall", "f1"]
    metric_coords   = list(range(len(metrics_to_plot)))
    # Use numeric y-coordinates so axis labels move with the heatmap cells during zoom/pan.
    fig = go.Figure(go.Heatmap(z=sweep_df[metrics_to_plot].values.T,
                               x=sweep_df["threshold"].tolist(), y=metric_coords,
                               colorscale="RdYlGn", zmin=0, zmax=1))
    fig.add_trace(go.Scatter(x=sweep_df["threshold"], y=sweep_df["support"] / len(y_arr),
                             mode="lines", name="Positive rate", yaxis="y2",
                             line=dict(color="navy", width=1.5, dash="dot")))
    fig.update_layout(title="Threshold Sweep — Precision / Recall / F1",
                      xaxis_title="Decision threshold",
                      yaxis=dict(title="Metric", tickvals=metric_coords, ticktext=metrics_to_plot,
                                 fixedrange=False),
                      yaxis2=dict(title="Predicted positive rate", overlaying="y", side="right",
                                  range=[0, 1], showgrid=False),
                      template="simple_white")
    path = PLOT_OUTPUT_DIR / "threshold_sweep.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    log.info("  Threshold sweep → %s", path)


def plot_outlier_report(
    pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_test: pd.Series, num_cols: list[str], use_pca: bool,
) -> None:
    if not num_cols:
        log.info("  Outlier report skipped (no numerical features)."); return
    log.info("  IsolationForest outlier detection (contamination=%.2f)...", OUTLIER_CONTAMINATION)
    prep      = pipeline.named_steps["preprocessor"]
    X_tr_proc = prep.transform(X_train)
    X_te_proc = prep.transform(X_test)
    iso = IsolationForest(contamination=OUTLIER_CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_tr_proc)
    scores = iso.decision_function(X_te_proc)
    is_out = iso.predict(X_te_proc) == -1
    n_out  = is_out.sum()
    log.info("  Flagged %d / %d test samples as suspected outliers.", n_out, len(is_out))
    if X_te_proc.shape[1] < 2:
        log.info("  Outlier report: not enough dimensions, skipping plot."); return
    x_vals, y_vals = X_te_proc[:, 0], X_te_proc[:, 1]
    x_lbl = "PC1" if use_pca else (num_cols[0] if num_cols else "Feature 0")
    y_lbl = "PC2" if use_pca else (num_cols[1] if len(num_cols) > 1 else "Feature 1")
    color_vals = np.where(is_out, "Outlier", "Normal")
    path = scatter_large_html(
        x_vals, y_vals,
        color=color_vals,
        title=f"IsolationForest - {n_out} outliers flagged (contamination={OUTLIER_CONTAMINATION:.2f})",
        x_label=x_lbl,
        y_label=y_lbl,
        output_name="outlier_report",
        color_label="Status",
        width=900,
        height=620,
    )
    log.info("  Outlier report → %s", path)
    _remove_stale_plot_variants("outlier_report", path)


def plot_partial_dependence(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series,
    num_cols: list[str], ohe_cat_cols: list[str], te_cat_cols: list[str],
    use_pca: bool, task: str,
) -> None:
    if use_pca:
        log.info("  PDP skipped (PCA active)."); return
    if not (num_cols + ohe_cat_cols + te_cat_cols):
        return

    prep       = pipeline.named_steps["preprocessor"]
    model_step = pipeline.named_steps["model"]
    base_model = model_step
    if hasattr(model_step, "calibrated_classifiers_"):
        base_model = model_step.calibrated_classifiers_[0].estimator
    elif hasattr(model_step, "estimator"):
        base_model = model_step.estimator

    feat_names: list[str] = list(num_cols)
    if ohe_cat_cols:
        ohe = prep.named_transformers_["cat"].named_steps["encoder"]
        feat_names.extend(ohe.get_feature_names_out(ohe_cat_cols).tolist())
    if te_cat_cols:
        feat_names.extend(te_cat_cols)

    scores  = base_model.feature_importances_
    n_use   = min(len(feat_names), len(scores))
    top_idx = np.argsort(scores[:n_use])[::-1][:PDP_TOP_N]
    top_names = [feat_names[i] for i in top_idx]
    log.info("  PDP/ICE for top-%d features: %s", PDP_TOP_N, top_names)

    X_proc = prep.transform(X_train)
    _MAX_ICE = 500
    rng = np.random.default_rng(RANDOM_STATE)
    X_ice = X_proc[rng.choice(X_proc.shape[0], size=min(_MAX_ICE, X_proc.shape[0]), replace=False)]

    try:
        from plotly.subplots import make_subplots

        n_cols = min(3, len(top_idx))
        n_rows = (len(top_idx) + n_cols - 1) // n_cols
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=top_names)
        for pos, (fi, fname) in enumerate(zip(top_idx, top_names), start=1):
            row = (pos - 1) // n_cols + 1
            col = (pos - 1) % n_cols + 1
            pd_result = partial_dependence(
                base_model, X_ice, features=[int(fi)], kind="both",
                grid_resolution=40,
            )
            grid = pd_result["grid_values"][0]
            avg = np.asarray(pd_result["average"])[0].reshape(-1)
            individual = np.asarray(pd_result["individual"])[0]
            if individual.ndim == 3:
                individual = individual[0]
            for line in individual[: min(120, individual.shape[0])]:
                fig.add_trace(go.Scatter(
                    x=grid, y=line, mode="lines",
                    line=dict(color="rgba(68,119,170,0.08)", width=0.7),
                    hoverinfo="skip", showlegend=False,
                ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=grid, y=avg, mode="lines", name=f"PDP - {fname}",
                line=dict(color="#CC3333", width=2.5), showlegend=False,
            ), row=row, col=col)
            fig.update_xaxes(title_text=fname, row=row, col=col)
            fig.update_yaxes(title_text="Partial dependence", row=row, col=col)
        fig.update_layout(
            title="PDP + ICE - top features",
            template="simple_white",
            height=max(420, 320 * n_rows),
        )
        path_html = PLOT_OUTPUT_DIR / "pdp_all.html"
        fig.write_html(str(path_html), include_plotlyjs="cdn")
        log.info("  PDP (all features) → %s", path_html)
    except Exception as e:
        log.warning("  PDP batch call failed (skipping): %s", e)


def _extract_booster_and_feature_names(
    pipeline: Pipeline,
    num_cols: list[str],
    ohe_cat_cols: list[str],
    te_cat_cols: list[str],
    use_pca: bool,
):
    """Shared helper: unwrap calibration/ensemble wrapper → raw XGBModel + feature names."""
    prep = pipeline.named_steps["preprocessor"]
    model_step = pipeline.named_steps["model"]

    # Unwrap CalibratedClassifierCV or similar wrappers
    base_model = model_step
    if hasattr(model_step, "calibrated_classifiers_"):
        base_model = model_step.calibrated_classifiers_[0].estimator
    elif hasattr(model_step, "estimator"):
        base_model = model_step.estimator

    booster = base_model.get_booster()

    # Build feature names matching the preprocessed matrix column order
    num_pipe = prep.named_transformers_.get("num")
    if use_pca and num_pipe is not None and "pca" in num_pipe.named_steps:
        pca = num_pipe.named_steps["pca"]
        n_comp = pca.n_components_ if hasattr(pca, "n_components_") else pca.components_.shape[0]
        feat_names: list[str] = [f"PC{i+1}" for i in range(n_comp)]
    else:
        feat_names = list(num_cols)

    if ohe_cat_cols:
        ohe = prep.named_transformers_["cat"].named_steps["encoder"]
        feat_names.extend(ohe.get_feature_names_out(ohe_cat_cols).tolist())
    if te_cat_cols:
        feat_names.extend(te_cat_cols)

    return booster, base_model, feat_names


def plot_shap_summary(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    num_cols: list[str],
    ohe_cat_cols: list[str],
    te_cat_cols: list[str],
    use_pca: bool,
    task: str,
    *,
    max_display: int = 20,
    max_samples: int = 2_000,
) -> None:
    """SHAP beeswarm summary plot using XGBoost's native contribution path.

    Extracts the booster from inside the sklearn Pipeline (unwrapping any
    CalibratedClassifierCV wrapper), computes contribution values on the
    preprocessed test matrix, then saves a raster beeswarm.
    """
    log.info("  Computing SHAP values (XGBoost pred_contribs)...")
    try:
        booster, base_model, feat_names = _extract_booster_and_feature_names(
            pipeline, num_cols, ohe_cat_cols, te_cat_cols, use_pca
        )
        prep = pipeline.named_steps["preprocessor"]
        X_proc = prep.transform(X_test)
        
        # Subsample for speed on large test sets
        rng = np.random.default_rng(RANDOM_STATE)
        n = X_proc.shape[0]
        if n > max_samples:
            idx = rng.choice(n, size=max_samples, replace=False)
            X_proc = X_proc[idx]

        sv = booster.predict(xgb.DMatrix(X_proc), pred_contribs=True)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        if sv.shape[1] > X_proc.shape[1]:
            sv = sv[:, :-1]  # drop bias term

        n_feat = min(len(feat_names), sv.shape[1])
        names_trimmed = feat_names[:n_feat]
        sv = sv[:, :n_feat]

        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)[::-1][:max_display]
        display_names = [names_trimmed[i] for i in order]
        display_sv = sv[:, order]

        # Normalise feature values for colouring (use raw processed data)
        X_display = X_proc[:, order[:n_feat]]
        feat_min = X_display.min(axis=0)
        feat_max = X_display.max(axis=0)
        feat_range = np.where(feat_max - feat_min == 0, 1.0, feat_max - feat_min)
        feat_norm = (X_display - feat_min) / feat_range  # 0–1 per feature

        rng2 = np.random.default_rng(RANDOM_STATE + 1)
        all_x: list[float] = []
        all_y: list[float] = []
        all_colors: list[float] = []
        for fi in range(len(display_names) - 1, -1, -1):  # bottom to top
            shap_col = display_sv[:, fi]
            color_col = feat_norm[:, fi] if fi < feat_norm.shape[1] else np.zeros(len(shap_col))
            jitter = rng2.uniform(-0.35, 0.35, size=len(shap_col))
            y_vals = (np.full(len(shap_col), fi) + jitter).tolist()
            all_x.extend(shap_col.tolist())
            all_y.extend(y_vals)
            all_colors.extend(color_col.tolist())

        path = scatter_large(
            all_x, all_y,
            color=all_colors,
            title=f"SHAP Summary — top {len(display_names)} features "
                  f"(n={X_proc.shape[0]:,}, native contributions)",
            x_label="SHAP value (impact on model output)",
            y_label="",
            output_name="shap_summary",
            color_label="Feature value (low to high)",
            y_tickvals=list(range(len(display_names))),
            y_ticktext=display_names,
            cmap="RdBu_r",
            width=950,
            height=max(440, 32 * len(display_names)),
        )
        log.info("  SHAP summary → %s", path)

    except Exception as e:
        log.warning("  SHAP summary failed (skipping): %s", e)


def plot_shap_interactions(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    num_cols: list[str],
    ohe_cat_cols: list[str],
    te_cat_cols: list[str],
    use_pca: bool,
    *,
    top_n: int = 10,
    max_samples: int = 500,
) -> None:
    """SHAP interaction heatmap (mean |SHAP interaction value| matrix).

    Interaction values are O(n² features) so we cap samples and features
    aggressively to keep runtime reasonable.
    """
    log.info("  Computing SHAP interaction values (native top-%d, n≤%d)...", top_n, max_samples)
    try:
        booster, _, feat_names = _extract_booster_and_feature_names(
            pipeline, num_cols, ohe_cat_cols, te_cat_cols, use_pca
        )
        prep = pipeline.named_steps["preprocessor"]
        X_proc = prep.transform(X_test)

        rng = np.random.default_rng(RANDOM_STATE)
        n = X_proc.shape[0]
        if n > max_samples:
            idx = rng.choice(n, size=max_samples, replace=False)
            X_proc = X_proc[idx]

        interaction_values = booster.predict(xgb.DMatrix(X_proc), pred_interactions=True)

        # Native multiclass shape is usually (n, classes, f+1, f+1); take class 1
        # when available and drop the bias row/column.
        if interaction_values.ndim == 4:
            cls_idx = 1 if interaction_values.shape[1] > 1 else 0
            interaction_values = interaction_values[:, cls_idx, :, :]
        if interaction_values.shape[1] > X_proc.shape[1]:
            interaction_values = interaction_values[:, :-1, :-1]

        n_feat = min(len(feat_names), interaction_values.shape[1])
        names_trimmed = feat_names[:n_feat]
        interaction_values = interaction_values[:, :n_feat, :n_feat]

        # Select top-N features by mean absolute main-effect (diagonal)
        main_effects = np.abs(interaction_values[:, range(n_feat), range(n_feat)]).mean(axis=0)
        top_idx = np.argsort(main_effects)[::-1][:min(top_n, n_feat)]
        top_names = [names_trimmed[i] for i in top_idx]

        mat = np.abs(interaction_values[:, top_idx, :][:, :, top_idx]).mean(axis=0)

        path = heatmap_large(
            mat,
            title=f"SHAP Interaction Values — top {len(top_names)} features "
                  f"(n={X_proc.shape[0]:,})",
            output_name="shap_interactions",
            x_ticktext=top_names,
            y_ticktext=top_names,
            cmap="Blues",
            value_label="Mean |interaction|",
            show_grid=True,
            width=max(620, 42 * len(top_names)),
            height=max(520, 42 * len(top_names)),
        )
        log.info("  SHAP interactions → %s", path)

    except Exception as e:
        log.warning("  SHAP interaction values failed (skipping): %s", e)


def plot_calibration_curve(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task: str,
    *,
    n_bins: int = 10,
) -> None:
    """Reliability diagram: calibrated vs uncalibrated predicted probabilities.

    Only runs for binary classification. Skipped silently for regression
    or multi-class tasks.
    """
    if task != "classification":
        return

    y_arr = np.asarray(y_val)
    if len(np.unique(y_arr)) != 2:
        log.info("  Calibration curve skipped (multi-class).")
        return

    log.info("  Plotting calibration curve...")
    try:
        from sklearn.calibration import calibration_curve

        model_step = pipeline.named_steps["model"]
        prep = pipeline.named_steps["preprocessor"]
        X_proc = prep.transform(X_val)

        traces = []

        # ── Calibrated (final pipeline model) ────────────────────────────────
        y_prob_cal = model_step.predict_proba(X_proc)[:, 1]
        frac_pos_cal, mean_pred_cal = calibration_curve(y_arr, y_prob_cal, n_bins=n_bins)
        traces.append(go.Scatter(
            x=mean_pred_cal, y=frac_pos_cal, mode="lines+markers",
            name="Calibrated", line=dict(color="#44AA77", width=2),
            marker=dict(size=7),
        ))

        # ── Uncalibrated: reach inside CalibratedClassifierCV if present ─────
        raw_xgb = None
        if hasattr(model_step, "calibrated_classifiers_"):
            raw_xgb = model_step.calibrated_classifiers_[0].estimator
        elif hasattr(model_step, "estimator"):
            raw_xgb = model_step.estimator

        if raw_xgb is not None and hasattr(raw_xgb, "predict_proba"):
            y_prob_raw = raw_xgb.predict_proba(X_proc)[:, 1]
            frac_pos_raw, mean_pred_raw = calibration_curve(y_arr, y_prob_raw, n_bins=n_bins)
            traces.append(go.Scatter(
                x=mean_pred_raw, y=frac_pos_raw, mode="lines+markers",
                name="Uncalibrated (raw XGBoost)",
                line=dict(color="#CC3333", width=2, dash="dot"),
                marker=dict(size=7),
            ))

        # ── Perfect calibration reference line ────────────────────────────────
        traces.append(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color="gray", dash="dash", width=1),
            name="Perfect calibration",
        ))

        fig = go.Figure(traces)
        fig.update_layout(
            title="Calibration Curve (Reliability Diagram)",
            xaxis_title="Mean predicted probability",
            yaxis_title="Fraction of positives",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
            template="simple_white",
            legend=dict(x=0.02, y=0.98),
        )
        path = PLOT_OUTPUT_DIR / "calibration_curve.html"
        fig.write_html(str(path), include_plotlyjs="cdn")
        log.info("  Calibration curve → %s", path)

    except Exception as e:
        log.warning("  Calibration curve failed (skipping): %s", e)


def plot_correlation_heatmap(
    X_train: pd.DataFrame,
    num_cols: list[str],
    *,
    max_features: int = 40,
) -> None:
    """Pearson correlation heatmap for numerical features.

    Capped at max_features to keep the plot readable; selects features
    with the highest mean absolute correlation with all other features.
    """
    cols = [c for c in num_cols if c in X_train.columns]
    if len(cols) < 2:
        log.info("  Correlation heatmap skipped (fewer than 2 numerical features).")
        return

    log.info("  Computing correlation heatmap (%d numerical features)...", len(cols))
    try:
        corr = X_train[cols].corr(method="pearson")

        # If more features than max_features, keep the most inter-correlated ones
        if len(cols) > max_features:
            mean_abs_corr = corr.abs().mean(axis=1)
            top_cols = mean_abs_corr.nlargest(max_features).index.tolist()
            corr = corr.loc[top_cols, top_cols]
            log.info("    Heatmap trimmed to top-%d most correlated features.", max_features)

        labels = corr.columns.tolist()
        z = corr.values

        z_display = z.copy().astype(float)
        z_display[np.triu_indices_from(z_display, k=1)] = np.nan

        path = heatmap_large_html(
            z_display,
            title=f"Feature Correlation Matrix ({len(labels)} numerical features)",
            output_name="corr_heatmap",
            x_ticktext=labels,
            y_ticktext=labels,
            cmap="RdBu_r",
            center=0.0,
            value_label="Pearson r",
            width=max(600, 32 * len(labels)),
            height=max(520, 32 * len(labels)),
        )
        log.info("  Correlation heatmap → %s", path)
        _remove_stale_plot_variants("corr_heatmap", path)

    except Exception as e:
        log.warning("  Correlation heatmap failed (skipping): %s", e)