"""plots.py — All Plotly and Matplotlib visualisation functions."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline

from .settings import (
    OUTLIER_CONTAMINATION, PCA_VARIANCE, PDP_TOP_N,
    PLOT_OUTPUT_DIR, RANDOM_STATE, TARGET_COL,
)

log = logging.getLogger(__name__)


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
    sorted_df = imp_df.sort_values("importance").reset_index(drop=True)
    coords    = list(range(len(sorted_df)))
    # Numeric y-axis so tick labels travel with bars during zoom/pan.
    fig = go.Figure(go.Bar(
        x=sorted_df["importance"], y=coords, orientation="h",
        marker=dict(color=sorted_df["importance"], colorscale="Blues", showscale=False),
    ))
    fig.update_layout(
        title="Top 20 Feature Importances — XGBoost (gain)",
        xaxis_title="Importance (gain)",
        yaxis=dict(tickvals=coords, ticktext=sorted_df["feature"].tolist(),
                   title="", fixedrange=False),
        template="simple_white",
    )
    path = PLOT_OUTPUT_DIR / "feature_importance.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    log.info("  Feature importance → %s", path)


def plot_pca_diagnostics(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, task: str,
) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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

    fig2, ax = plt.subplots(figsize=(7, 5))
    if task == "regression":
        sc = ax.scatter(pc1, pc2, c=y_train.values, cmap="viridis", alpha=0.65, s=18, linewidths=0)
        plt.colorbar(sc, ax=ax).set_label(TARGET_COL, fontsize=9)
    else:
        import matplotlib
        classes = np.unique(y_train.values)
        palette = matplotlib.colormaps["tab10"].resampled(len(classes))
        for idx, cls in enumerate(classes):
            mask = y_train.values == cls
            ax.scatter(pc1[mask], pc2[mask], color=palette(idx), alpha=0.65, s=18, linewidths=0, label=str(cls))
        ax.legend(title="Class", fontsize=8, markerscale=1.5)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("PCA — first two principal components")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path2 = PLOT_OUTPUT_DIR / "pca_2d.png"
    plt.savefig(str(path2), dpi=100, bbox_inches="tight"); plt.close()
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
    log.info("  Computing permutation importance (n_repeats=10)...")
    scoring_map = {"roc_auc": "roc_auc", "auprc": "average_precision",
                   "macro_f1": "f1_macro", "weighted_f1": "f1_weighted", "r2": "r2"}
    scoring = scoring_map.get(metric.name, "r2")

    if X_test_proc is not None:
        result = permutation_importance(
            pipeline.named_steps["model"], X_test_proc, y_test,
            scoring=scoring, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
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
            scoring=scoring, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
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
    log.info("  Computing learning curve (cv=3, 8 sizes)...")
    scoring_map = {"roc_auc": "roc_auc", "auprc": "average_precision",
                   "macro_f1": "f1_macro", "weighted_f1": "f1_weighted", "r2": "r2"}
    scoring = scoring_map.get(metric.name, "r2")
    train_sizes_frac = np.logspace(np.log10(0.10), 0, num=8)
    cv_lc = (StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
             if task == "classification" else 3)

    # If pre-processed data is provided, run learning_curve on just the model
    # step to avoid re-running the preprocessor on every fold (same output,
    # significantly faster).
    if X_trainval_proc is not None:
        estimator = pipeline.named_steps["model"]
        X_lc = X_trainval_proc
    else:
        estimator = pipeline
        X_lc = X_trainval

    try:
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X_lc, y_trainval,
            train_sizes=train_sizes_frac, cv=cv_lc, scoring=scoring, n_jobs=-1, shuffle=False,
        )
    except Exception as e:
        log.warning("  Learning curve failed (skipping): %s", e); return

    tr_mean = train_scores.mean(axis=1); tr_std = train_scores.std(axis=1)
    vl_mean = val_scores.mean(axis=1);  vl_std = val_scores.std(axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes.tolist(), y=tr_mean.tolist(), mode="lines+markers",
                             name="Train", line=dict(color="#4477AA", width=2),
                             error_y=dict(type="data", array=tr_std.tolist(), visible=True)))
    fig.add_trace(go.Scatter(x=train_sizes.tolist(), y=vl_mean.tolist(), mode="lines+markers",
                             name="Val (CV-3)", line=dict(color="#CC3333", width=2),
                             error_y=dict(type="data", array=vl_std.tolist(), visible=True)))
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
    color_vals = ["Outlier" if o else "Normal" for o in is_out]
    fig = px.scatter(x=x_vals, y=y_vals, color=color_vals,
                     color_discrete_map={"Normal": "#4477AA", "Outlier": "#CC3333"},
                     hover_data={"Anomaly Score": np.round(scores, 4)},
                     labels={"x": x_lbl, "y": y_lbl, "color": "Status"},
                     title=f"IsolationForest — {n_out} outliers flagged "
                           f"(contamination={OUTLIER_CONTAMINATION:.2f})",
                     opacity=0.55, template="simple_white")
    fig.update_traces(marker=dict(size=5))
    path = PLOT_OUTPUT_DIR / "outlier_report.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    log.info("  Outlier report → %s", path)


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

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Compute all PDPs in one batched call. sklearn internally vectorises the
    # grid evaluations across features, which is significantly faster than
    # calling from_estimator once per feature in a Python loop.
    try:
        n_cols = min(3, len(top_idx))
        n_rows = (len(top_idx) + n_cols - 1) // n_cols
        fig_mpl, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(7 * n_cols, 4 * n_rows),
            squeeze=False,
        )
        PartialDependenceDisplay.from_estimator(
            base_model, X_ice,
            features=[int(fi) for fi in top_idx],
            feature_names=feat_names,
            kind="both",
            subsample=_MAX_ICE,
            random_state=RANDOM_STATE,
            ax=axes.ravel()[:len(top_idx)],
            pd_line_kw={"color": "#CC3333", "linewidth": 2.5},
            ice_lines_kw={"color": "#4477AA", "alpha": 0.06, "linewidth": 0.8},
        )
        # Hide any unused axes in the grid
        for ax in axes.ravel()[len(top_idx):]:
            ax.set_visible(False)
        for ax, fname in zip(axes.ravel(), top_names):
            ax.set_title(f"PDP + ICE — {fname}")
            ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        path_png = PLOT_OUTPUT_DIR / "pdp_all.png"
        plt.savefig(str(path_png), dpi=150, bbox_inches="tight")
        plt.close()
        log.info("  PDP (all features) → %s", path_png)
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
    """SHAP beeswarm summary plot using TreeExplainer on the raw booster.

    Extracts the booster from inside the sklearn Pipeline (unwrapping any
    CalibratedClassifierCV wrapper), runs SHAP on the preprocessed test
    matrix, then saves an interactive Plotly HTML beeswarm.
    """
    try:
        import shap  # noqa: F401
    except ImportError:
        log.warning("  [shap] shap not installed — skipping SHAP plots. "
                    "Install with: pip install shap")
        return

    log.info("  Computing SHAP values (TreeExplainer)...")
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

        explainer = shap.TreeExplainer(booster)
        shap_values = explainer(X_proc)

        # For multiclass shap returns shape (n, features, classes) — take class-1 slice
        sv = shap_values.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        n_feat = min(len(feat_names), sv.shape[1])
        names_trimmed = feat_names[:n_feat]
        sv = sv[:, :n_feat]

        # Build beeswarm data: one point per (sample, feature), jittered on y
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

        # Build a single consolidated scatter trace using None separators between
        # features. This is much faster than adding one trace per feature because
        # Plotly rebuilds internal state on every add_trace call.
        rng2 = np.random.default_rng(RANDOM_STATE + 1)
        all_x: list = []
        all_y: list = []
        all_colors: list = []
        all_hover: list = []
        for fi in range(len(display_names) - 1, -1, -1):  # bottom to top
            shap_col = display_sv[:, fi]
            color_col = feat_norm[:, fi] if fi < feat_norm.shape[1] else np.zeros(len(shap_col))
            jitter = rng2.uniform(-0.35, 0.35, size=len(shap_col))
            y_vals = (np.full(len(shap_col), fi) + jitter).tolist()
            all_x.extend(shap_col.tolist() + [None])
            all_y.extend(y_vals + [None])
            all_colors.extend(color_col.tolist() + [0.0])
            all_hover.extend([f"<b>{display_names[fi]}</b><br>SHAP: {v:.4f}<extra></extra>"
                               for v in shap_col] + [None])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=all_x,
            y=all_y,
            mode="markers",
            marker=dict(
                color=all_colors,
                colorscale="RdBu_r",
                cmin=0, cmax=1,
                size=4, opacity=0.7,
                colorbar=dict(
                    title="Feature value<br>(low → high)",
                    thickness=12, len=0.5,
                    tickvals=[0, 1], ticktext=["Low", "High"],
                ),
                showscale=True,
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=all_hover,
            showlegend=False,
        ))

        fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
        fig.update_layout(
            title=f"SHAP Summary — top {len(display_names)} features "
                  f"(n={X_proc.shape[0]:,}, TreeExplainer)",
            xaxis_title="SHAP value (impact on model output)",
            yaxis=dict(
                tickvals=list(range(len(display_names))),
                ticktext=display_names,
                title="",
                fixedrange=False,
            ),
            template="simple_white",
            height=max(400, 30 * len(display_names)),
        )
        path = PLOT_OUTPUT_DIR / "shap_summary.html"
        fig.write_html(str(path), include_plotlyjs="cdn")
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
    try:
        import shap  # noqa: F401
    except ImportError:
        return  # already warned in plot_shap_summary

    log.info("  Computing SHAP interaction values (top-%d features, n≤%d)...", top_n, max_samples)
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

        explainer = shap.TreeExplainer(booster)
        # shap_interaction_values shape: (n_samples, n_features, n_features)
        interaction_values = explainer.shap_interaction_values(X_proc)

        # For multiclass it returns a list — take class-1
        if isinstance(interaction_values, list):
            interaction_values = interaction_values[1]

        # Trim to available feature names
        n_feat = min(len(feat_names), interaction_values.shape[1])
        names_trimmed = feat_names[:n_feat]
        interaction_values = interaction_values[:, :n_feat, :n_feat]

        # Select top-N features by mean absolute main-effect (diagonal)
        main_effects = np.abs(interaction_values[:, range(n_feat), range(n_feat)]).mean(axis=0)
        top_idx = np.argsort(main_effects)[::-1][:min(top_n, n_feat)]
        top_names = [names_trimmed[i] for i in top_idx]

        mat = np.abs(interaction_values[:, top_idx, :][:, :, top_idx]).mean(axis=0)

        fig = go.Figure(go.Heatmap(
            z=mat,
            x=top_names,
            y=top_names,
            colorscale="Blues",
            colorbar=dict(title="Mean |interaction|", thickness=14),
            hovertemplate="<b>%{y} × %{x}</b><br>Mean |interaction|: %{z:.4f}<extra></extra>",
        ))
        fig.update_layout(
            title=f"SHAP Interaction Values — top {len(top_names)} features "
                  f"(n={X_proc.shape[0]:,})",
            xaxis=dict(tickangle=-35, fixedrange=False),
            yaxis=dict(autorange="reversed", fixedrange=False),
            template="simple_white",
            height=max(450, 35 * len(top_names)),
        )
        path = PLOT_OUTPUT_DIR / "shap_interactions.html"
        fig.write_html(str(path), include_plotlyjs="cdn")
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

        # Mask upper triangle so we don't double-display (set to None)
        z_display = z.copy().astype(object)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                z_display[i, j] = None

        fig = go.Figure(go.Heatmap(
            z=z_display,
            x=labels,
            y=labels,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1, zmax=1,
            colorbar=dict(title="Pearson r", thickness=14),
            hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
            text=np.where(z_display == None, "", np.round(z_display.astype(float), 2)),  # noqa: E711
            texttemplate="%{text}",
            textfont=dict(size=9),
        ))
        fig.update_layout(
            title=f"Feature Correlation Matrix ({len(labels)} numerical features)",
            xaxis=dict(tickangle=-40, fixedrange=False),
            yaxis=dict(autorange="reversed", fixedrange=False),
            template="simple_white",
            height=max(450, 28 * len(labels)),
            width=max(500, 30 * len(labels)),
        )
        path = PLOT_OUTPUT_DIR / "corr_heatmap.html"
        fig.write_html(str(path), include_plotlyjs="cdn")
        log.info("  Correlation heatmap → %s", path)

    except Exception as e:
        log.warning("  Correlation heatmap failed (skipping): %s", e)