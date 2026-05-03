"""features.py — Feature type detection, variance filter, interactions, RFECV."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

from .settings import (
    CARDINALITY_LIMIT, TARGET_ENC_THRESHOLD, RANDOM_STATE,
    VARIANCE_THRESHOLD, INTERACTION_TOP_K,
)
from .metrics import MetricConfig

log = logging.getLogger(__name__)


def detect_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """UPGRADE 3/16/24: Infer (num_cols, ohe_cat_cols, te_cat_cols)."""
    all_nan = X.columns[X.isnull().all()].tolist()
    if all_nan:
        log.warning("  Dropping %d all-NaN columns: %s", len(all_nan), all_nan)
        X = X.drop(columns=all_nan)

    explicit_cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    ohe_explicit, te_explicit = [], []
    for col in explicit_cat:
        n_unique = X[col].nunique()
        if n_unique > TARGET_ENC_THRESHOLD:
            te_explicit.append(col)
            log.info("  Auto-routing '%s' → TargetEncoder (cardinality=%d > %d)",
                     col, n_unique, TARGET_ENC_THRESHOLD)
        else:
            ohe_explicit.append(col)

    int_cols = X.select_dtypes(include=["int64", "int32"]).columns.tolist()
    low_card_int, high_card_int = [], []
    for col in int_cols:
        n_unique = X[col].nunique()
        if n_unique <= CARDINALITY_LIMIT:
            low_card_int.append(col)
            log.info("  Auto-promoting '%s' → categorical (int, cardinality=%d ≤ %d)",
                     col, n_unique, CARDINALITY_LIMIT)
        else:
            high_card_int.append(col)

    float_cols   = X.select_dtypes(include=["float64", "float32"]).columns.tolist()
    num_cols     = high_card_int + float_cols
    ohe_cat_cols = ohe_explicit + low_card_int
    te_cat_cols  = te_explicit

    log.info("[4/9] Features detected — num=%d, ohe_cat=%d, te_cat=%d",
             len(num_cols), len(ohe_cat_cols), len(te_cat_cols))
    return num_cols, ohe_cat_cols, te_cat_cols


def filter_low_variance(
    X_train: pd.DataFrame,
    num_cols: list[str],
    ohe_cat_cols: list[str],
    te_cat_cols: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """N3: Drop near-constant numerical features via VarianceThreshold."""
    if not num_cols:
        return num_cols, ohe_cat_cols, te_cat_cols

    log.info("[VarFilter] VarianceThreshold(threshold=%.4f) on %d numerical cols...",
             VARIANCE_THRESHOLD, len(num_cols))
    imputer = SimpleImputer(strategy="median")
    X_num   = imputer.fit_transform(X_train[num_cols])
    vt      = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
    vt.fit(X_num)
    support = vt.get_support()
    kept    = [c for c, k in zip(num_cols, support) if k]
    dropped = [c for c, k in zip(num_cols, support) if not k]
    if dropped:
        variances = X_num.var(axis=0)
        for col, keep, var in zip(num_cols, support, variances):
            if not keep:
                log.info("  VarFilter: dropping '%s' (variance=%.6g)", col, float(var))
        log.info("  VarFilter removed %d feature(s): %s", len(dropped), dropped)
    else:
        log.info("  VarFilter: all %d features pass.", len(num_cols))
    return kept, ohe_cat_cols, te_cat_cols


def generate_feature_interactions(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    num_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """N8: Add top-K pairwise correlation interaction terms."""
    if len(num_cols) < 2:
        log.info("  [InteractionGen] Skipped — fewer than 2 numerical features.")
        return X_train, X_val, X_test, num_cols

    log.info("[InteractionGen] top_k=%d on %d numerical cols...", INTERACTION_TOP_K, len(num_cols))

    imputer   = SimpleImputer(strategy="median")
    X_num_imp = pd.DataFrame(
        imputer.fit_transform(X_train[num_cols]),
        columns=num_cols, index=X_train.index,
    )
    corr = X_num_imp.corr(method="pearson")
    cols_arr = corr.columns.tolist()
    pairs: list[tuple[float, str, str]] = []
    for i in range(len(cols_arr)):
        for j in range(i + 1, len(cols_arr)):
            r = corr.iloc[i, j]
            if np.isfinite(r):
                pairs.append((abs(float(r)), cols_arr[i], cols_arr[j]))
    pairs.sort(key=lambda x: x[0], reverse=True)
    selected = [(r, a, b) for r, a, b in pairs if r >= 0.05][:INTERACTION_TOP_K]

    if not selected:
        log.info("  [InteractionGen] No pairs with |corr| ≥ 0.05. Skipped.")
        return X_train, X_val, X_test, num_cols

    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
    new_cols: list[str] = []
    for rank, (r, col_a, col_b) in enumerate(selected, 1):
        name = f"{col_a}__x__{col_b}"
        log.info("    %2d. %-40s |r|=%.4f", rank, name, r)
        X_train[name] = X_train[col_a] * X_train[col_b]
        X_val[name]   = X_val[col_a]   * X_val[col_b]
        X_test[name]  = X_test[col_a]  * X_test[col_b]
        new_cols.append(name)

    extended = list(num_cols) + new_cols
    log.info("  [InteractionGen] Added %d interaction feature(s) → num_cols: %d → %d",
             len(new_cols), len(num_cols), len(extended))
    return X_train, X_val, X_test, extended


def select_features_rfecv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    num_cols: list[str],
    ohe_cat_cols: list[str],
    te_cat_cols: list[str],
    task: str,
    metric: MetricConfig,
) -> tuple[list[str], list[str], list[str]]:
    """U23: RFECV on a lightweight XGB (100 trees, depth=3, cv=3)."""
    all_cols = num_cols + ohe_cat_cols + te_cat_cols
    if len(all_cols) < 5:
        log.info("  RFECV skipped (fewer than 5 features).")
        return num_cols, ohe_cat_cols, te_cat_cols

    log.info("[RFECV] Running recursive feature elimination (cv=3)...")
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    all_cat = ohe_cat_cols + te_cat_cols
    transformers = []
    if num_cols:  transformers.append(("num", num_pipe, num_cols))
    if all_cat:   transformers.append(("cat", cat_pipe, all_cat))
    prep = ColumnTransformer(transformers=transformers, remainder="drop")

    rfe_model = (
        XGBClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE,
                      nthread=1, eval_metric=metric.eval_metric)
        if task == "classification"
        else XGBRegressor(n_estimators=100, max_depth=3, random_state=RANDOM_STATE,
                          nthread=1, eval_metric=metric.eval_metric)
    )
    scoring = {"roc_auc": "roc_auc", "auprc": "average_precision",
               "macro_f1": "f1_macro", "weighted_f1": "f1_weighted", "r2": "r2"}.get(metric.name, "r2")
    selector = RFECV(estimator=rfe_model, step=1, cv=3,
                     scoring=scoring, min_features_to_select=3, n_jobs=-1)
    X_rfe = prep.fit_transform(X_train, y_train)
    selector.fit(X_rfe, y_train)
    support = selector.support_

    ohe_feat_names: list[str] = list(num_cols)
    if all_cat:
        enc = prep.named_transformers_["cat"].named_steps["encoder"]
        ohe_feat_names.extend(enc.get_feature_names_out(all_cat).tolist())
    supported_names = set(np.array(ohe_feat_names)[support])

    new_num = [c for c in num_cols     if c in supported_names]
    new_ohe = [c for c in ohe_cat_cols if c in supported_names]
    new_te  = [c for c in te_cat_cols  if c in supported_names]

    dropped = (set(num_cols) - set(new_num)) | (set(ohe_cat_cols) - set(new_ohe)) | (set(te_cat_cols) - set(new_te))
    log.info("  RFECV: %d / %d features kept. Dropped: %s",
             support.sum(), len(support), sorted(dropped) if dropped else "none")
    return new_num, new_ohe, new_te