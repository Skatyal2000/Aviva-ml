import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

NUMERIC_FEATURES = ["invoice_amount", "estimate_amount", "approved_amount", "paid_amount", "vehicle_age", "liability_percentage", "repair_duration_days", "vehicle_claim_count", "invoice_vs_estimate_ratio", "invoice_vs_approved_ratio"]
CATEGORICAL_FEATURES = ["make", "model", "damage_severity", "claim_severity", "incident_state", "loss_cause", "repair_shop_contact_id"]

REG_NUMERIC_FEATURES = ["estimate_amount", "vehicle_age", "liability_percentage", "repair_duration_days", "vehicle_claim_count", "damage_severity_ordinal", "shop_avg_invoice"]
REG_CATEGORICAL_FEATURES = ["make", "model", "claim_severity", "incident_state", "loss_cause"]
SEVERITY_ORDER = {"Minor": 0, "Moderate": 1, "Severe": 2, "Total Loss": 3}


def build_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def fit_kmeans(df, n_clusters=6, random_state=42):
    pre = build_preprocessor()
    X = pre.fit_transform(df)
    X = X.toarray() if hasattr(X, "toarray") else X

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    distances = np.linalg.norm(X - kmeans.cluster_centers_[labels], axis=1)

    flag = np.zeros(len(df), dtype=bool)
    cluster_thresholds = {}
    for c in np.unique(labels):
        mask = labels == c
        thr = np.quantile(distances[mask], 0.95)
        cluster_thresholds[c] = thr
        flag[mask & (distances > thr)] = True

    return {"preprocessor": pre, "X": X, "model": kmeans, "labels": labels,
            "distances": distances, "cluster_thresholds": cluster_thresholds, "flag": flag}


def fit_dbscan(df, eps=2.9, min_samples=8):
    pre = build_preprocessor()
    X = pre.fit_transform(df)
    X = X.toarray() if hasattr(X, "toarray") else X

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    flag = labels == -1

    return {"preprocessor": pre, "X": X, "model": dbscan, "labels": labels, "flag": flag}


def fit_gmm(df, n_components=6, flag_percentile=5, random_state=42):
    pre = build_preprocessor()
    X = pre.fit_transform(df)
    X = X.toarray() if hasattr(X, "toarray") else X

    gmm = GaussianMixture(n_components=n_components, covariance_type="diag", random_state=random_state, n_init=5)
    gmm.fit(X)
    labels = gmm.predict(X)
    log_likelihood = gmm.score_samples(X)
    threshold = np.percentile(log_likelihood, flag_percentile)
    flag = log_likelihood < threshold

    return {"preprocessor": pre, "X": X, "model": gmm, "labels": labels,
            "log_likelihood": log_likelihood, "threshold": threshold, "flag": flag}


def _build_regression_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), REG_NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), REG_CATEGORICAL_FEATURES),
    ])


def fit_regression(df, flag_pct=95, random_state=42):
    """Predict invoice_amount, residual = anomaly score.

    Feature engineering: damage_severity -> ordinal, repair_shop_contact_id ->
    shop's historical average invoice (instead of a 220-column one-hot). The
    shop average is a target-derived feature, so it's computed from training
    data only in every fold below to avoid leakage.

    log1p(estimate_amount) was tried and dropped: identical results on tree
    models (monotonic transforms don't change tree splits) and it actively
    hurt Ridge (breaks the near-linear invoice ~ estimate relationship).
    """
    target = "invoice_amount"
    df = df.copy()
    df["damage_severity_ordinal"] = df["damage_severity"].map(SEVERITY_ORDER)

    oof_pred = np.zeros(len(df))
    for train_idx, val_idx in KFold(5, shuffle=True, random_state=random_state).split(df):
        tr, va = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
        shop_avg = tr.groupby("repair_shop_contact_id")[target].mean()
        global_avg = tr[target].mean()
        tr["shop_avg_invoice"] = tr["repair_shop_contact_id"].map(shop_avg)
        va["shop_avg_invoice"] = va["repair_shop_contact_id"].map(shop_avg).fillna(global_avg)

        fold_model = Pipeline([("pre", _build_regression_preprocessor()), ("gbm", HistGradientBoostingRegressor(random_state=random_state))])
        fold_model.fit(tr[REG_NUMERIC_FEATURES + REG_CATEGORICAL_FEATURES], tr[target])
        oof_pred[val_idx] = fold_model.predict(va[REG_NUMERIC_FEATURES + REG_CATEGORICAL_FEATURES])

    residual = df[target].values - oof_pred
    threshold = np.percentile(np.abs(residual), flag_pct)
    flag = np.abs(residual) > threshold

    # final model + full-history shop averages, for scoring new points later
    # (using full history here is correct, not leakage -- a new point was never part of it)
    shop_avg_full = df.groupby("repair_shop_contact_id")[target].mean()
    global_avg_full = df[target].mean()
    df["shop_avg_invoice"] = df["repair_shop_contact_id"].map(shop_avg_full)
    model = Pipeline([("pre", _build_regression_preprocessor()), ("gbm", HistGradientBoostingRegressor(random_state=random_state))])
    model.fit(df[REG_NUMERIC_FEATURES + REG_CATEGORICAL_FEATURES], df[target])

    return {"model": model, "oof_pred": oof_pred, "residual": residual,
            "threshold": threshold, "flag": flag,
            "shop_avg_full": shop_avg_full, "global_avg_full": global_avg_full}
