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
    AUTO_FE_ENABLED, AUTO_FE_ENGINE, AUTO_FE_MAX_FEATURES, AUTO_FE_MAX_DEPTH,
    AUTO_FE_ENTITY_ID_COL, AUTO_FE_TIME_INDEX_COL,
    AUTO_FE_TSFRESH_COLUMN_ID, AUTO_FE_TSFRESH_COLUMN_SORT,
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


def apply_auto_feature_engineering(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    num_cols: list[str],
    ohe_cat_cols: list[str],
    te_cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Opt-in automated feature generation with Featuretools or tsfresh.

    The generated columns are appended as numerical features. The function is
    intentionally conservative: it caps generated columns and skips cleanly when
    an optional backend is unavailable or the data does not fit that backend.
    """
    if not AUTO_FE_ENABLED:
        log.info("  [AutoFE] skipped — auto_feature_engineering.enabled=false")
        return X_train, X_val, X_test, num_cols

    engine = AUTO_FE_ENGINE.lower()
    if engine == "featuretools":
        return _apply_featuretools(
            X_train, X_val, X_test, num_cols, ohe_cat_cols + te_cat_cols
        )
    if engine == "tsfresh":
        return _apply_tsfresh(X_train, X_val, X_test, y_train, num_cols)

    log.warning("  [AutoFE] unknown engine='%s' — skipped", AUTO_FE_ENGINE)
    return X_train, X_val, X_test, num_cols


def _featuretools_entityset(X: pd.DataFrame, ft):
    frame = X.reset_index(drop=True).copy()
    row_id = "__autofe_row_id"
    if row_id in frame.columns:
        row_id = "__autofe_row_id__"
    frame[row_id] = np.arange(len(frame))
    es = ft.EntitySet(id="xgb_prototype_autofe")
    kwargs = {"dataframe_name": "observations", "dataframe": frame, "index": row_id}
    if AUTO_FE_TIME_INDEX_COL and AUTO_FE_TIME_INDEX_COL in frame.columns:
        kwargs["time_index"] = AUTO_FE_TIME_INDEX_COL
    es = es.add_dataframe(**kwargs)
    return es


def _clean_generated_features(
    matrix: pd.DataFrame,
    original_cols: set[str],
    prefix: str,
    keep_raw: list[str] | None = None,
) -> pd.DataFrame:
    generated = matrix.drop(columns=[c for c in matrix.columns if c in original_cols], errors="ignore")
    generated = generated.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
    if generated.empty:
        return generated
    if keep_raw is None:
        variances = generated.var(numeric_only=True).sort_values(ascending=False)
        keep = [c for c in variances.index if variances.loc[c] > 0][:AUTO_FE_MAX_FEATURES]
    else:
        keep = [c for c in keep_raw if c in generated.columns]
    generated = generated[keep].copy()
    generated.columns = [f"{prefix}{str(c).replace(' ', '_')[:80]}" for c in generated.columns]
    return generated


def _apply_featuretools(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    try:
        import featuretools as ft
    except ImportError:
        log.warning("  [AutoFE] featuretools not installed — skipped.")
        return X_train, X_val, X_test, num_cols

    numeric_source = [c for c in num_cols if c in X_train.columns]
    if len(numeric_source) < 2:
        log.info("  [AutoFE] Featuretools skipped — fewer than 2 numerical columns.")
        return X_train, X_val, X_test, num_cols

    log.info(
        "[AutoFE] Featuretools enabled (max_depth=%d, max_features=%d)...",
        AUTO_FE_MAX_DEPTH, AUTO_FE_MAX_FEATURES,
    )
    try:
        primitives = ["add_numeric", "multiply_numeric"]
        original_cols = set(X_train.columns)
        train_matrix, feature_defs = ft.dfs(
            entityset=_featuretools_entityset(X_train, ft),
            target_dataframe_name="observations",
            trans_primitives=primitives,
            agg_primitives=[],
            max_depth=AUTO_FE_MAX_DEPTH,
            features_only=False,
            verbose=False,
        )
        raw_generated = train_matrix.drop(
            columns=[c for c in train_matrix.columns if c in original_cols],
            errors="ignore",
        ).select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
        raw_variances = raw_generated.var(numeric_only=True).sort_values(ascending=False)
        keep_raw = [c for c in raw_variances.index if raw_variances.loc[c] > 0][:AUTO_FE_MAX_FEATURES]
        train_gen = _clean_generated_features(train_matrix, original_cols, "ft__", keep_raw=keep_raw)
        if train_gen.empty:
            log.info("  [AutoFE] Featuretools produced no usable numeric features.")
            return X_train, X_val, X_test, num_cols

        def _calc(X: pd.DataFrame) -> pd.DataFrame:
            matrix = ft.calculate_feature_matrix(
                features=feature_defs,
                entityset=_featuretools_entityset(X, ft),
                verbose=False,
            )
            gen = _clean_generated_features(matrix, original_cols, "ft__", keep_raw=keep_raw)
            return gen.reindex(columns=train_gen.columns)

        val_gen = _calc(X_val)
        test_gen = _calc(X_test)
    except Exception as exc:
        log.warning("  [AutoFE] Featuretools failed (%s) — skipped.", exc)
        return X_train, X_val, X_test, num_cols

    X_train_out = pd.concat([X_train.reset_index(drop=True), train_gen.reset_index(drop=True)], axis=1)
    X_val_out = pd.concat([X_val.reset_index(drop=True), val_gen.reset_index(drop=True)], axis=1)
    X_test_out = pd.concat([X_test.reset_index(drop=True), test_gen.reset_index(drop=True)], axis=1)
    new_cols = train_gen.columns.tolist()
    log.info("  [AutoFE] Added %d Featuretools feature(s): %s", len(new_cols), new_cols[:10])
    return X_train_out, X_val_out, X_test_out, num_cols + new_cols


def _apply_tsfresh(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    num_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    try:
        from tsfresh import extract_features
        from tsfresh.feature_extraction import MinimalFCParameters
    except ImportError:
        log.warning("  [AutoFE] tsfresh not installed — skipped.")
        return X_train, X_val, X_test, num_cols

    id_col = AUTO_FE_TSFRESH_COLUMN_ID or AUTO_FE_ENTITY_ID_COL
    sort_col = AUTO_FE_TSFRESH_COLUMN_SORT or AUTO_FE_TIME_INDEX_COL
    if not id_col or id_col not in X_train.columns:
        log.warning("  [AutoFE] tsfresh requires tsfresh_column_id/entity_id_col — skipped.")
        return X_train, X_val, X_test, num_cols

    value_cols = [c for c in num_cols if c in X_train.columns and c not in {id_col, sort_col}]
    if not value_cols:
        log.warning("  [AutoFE] tsfresh found no numeric value columns — skipped.")
        return X_train, X_val, X_test, num_cols

    def _extract(X: pd.DataFrame) -> pd.DataFrame:
        cols = [id_col] + ([sort_col] if sort_col and sort_col in X.columns else []) + value_cols
        feats = extract_features(
            X[cols],
            column_id=id_col,
            column_sort=sort_col if sort_col in X.columns else None,
            default_fc_parameters=MinimalFCParameters(),
            disable_progressbar=True,
            n_jobs=0,
        )
        feats = feats.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
        keep = feats.var(numeric_only=True).sort_values(ascending=False).head(AUTO_FE_MAX_FEATURES).index
        feats = feats[keep].add_prefix("tsfresh__")
        return feats

    try:
        train_gen = _extract(X_train)
        val_gen = _extract(X_val).reindex(columns=train_gen.columns)
        test_gen = _extract(X_test).reindex(columns=train_gen.columns)
    except Exception as exc:
        log.warning("  [AutoFE] tsfresh failed (%s) — skipped.", exc)
        return X_train, X_val, X_test, num_cols

    key_train = X_train[id_col].reset_index(drop=True)
    key_val = X_val[id_col].reset_index(drop=True)
    key_test = X_test[id_col].reset_index(drop=True)
    X_train_out = pd.concat([X_train.reset_index(drop=True), train_gen.reindex(key_train).reset_index(drop=True)], axis=1)
    X_val_out = pd.concat([X_val.reset_index(drop=True), val_gen.reindex(key_val).reset_index(drop=True)], axis=1)
    X_test_out = pd.concat([X_test.reset_index(drop=True), test_gen.reindex(key_test).reset_index(drop=True)], axis=1)
    new_cols = train_gen.columns.tolist()
    log.info("  [AutoFE] Added %d tsfresh feature(s): %s", len(new_cols), new_cols[:10])
    return X_train_out, X_val_out, X_test_out, num_cols + new_cols


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
