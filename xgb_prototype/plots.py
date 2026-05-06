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
    plt.savefig(str(path2), dpi=150, bbox_inches="tight"); plt.close()
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
) -> None:
    log.info("  Computing learning curve (cv=3, 8 sizes)...")
    scoring_map = {"roc_auc": "roc_auc", "auprc": "average_precision",
                   "macro_f1": "f1_macro", "weighted_f1": "f1_weighted", "r2": "r2"}
    scoring = scoring_map.get(metric.name, "r2")
    train_sizes_frac = np.logspace(np.log10(0.10), 0, num=8)
    cv_lc = (StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
             if task == "classification" else 3)
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            pipeline, X_trainval, y_trainval,
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

    for fi, fname in zip(top_idx, top_names):
        try:
            fig_mpl, ax = plt.subplots(figsize=(7, 4))
            PartialDependenceDisplay.from_estimator(
                base_model, X_ice, features=[int(fi)], feature_names=feat_names,
                kind="both", subsample=_MAX_ICE, random_state=RANDOM_STATE, ax=ax,
                pd_line_kw={"color": "#CC3333", "linewidth": 2.5},
                ice_lines_kw={"color": "#4477AA", "alpha": 0.06, "linewidth": 0.8},
            )
            ax.set_title(f"PDP + ICE — {fname}")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            safe = fname.replace(" ", "_").replace("/", "_")[:60]
            path_png = PLOT_OUTPUT_DIR / f"pdp_{safe}.png"
            plt.savefig(str(path_png), dpi=150, bbox_inches="tight"); plt.close()
            log.info("    PDP → %s", path_png)
        except Exception as e:
            log.warning("    PDP failed for '%s': %s", fname, e)