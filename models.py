import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold

NUMERIC_FEATURES = ["invoice_amount", "estimate_amount", "approved_amount", "paid_amount", "vehicle_age", "liability_percentage", "repair_duration_days", "vehicle_claim_count", "invoice_vs_estimate_ratio", "invoice_vs_approved_ratio"]
CATEGORICAL_FEATURES = ["make", "model", "damage_severity", "claim_severity", "incident_state", "loss_cause", "repair_shop_contact_id"]

REG_NUMERIC_FEATURES = ["estimate_amount", "vehicle_age", "liability_percentage", "repair_duration_days", "vehicle_claim_count"]
REG_CATEGORICAL_FEATURES = ["make", "model", "damage_severity", "claim_severity", "incident_state", "loss_cause", "repair_shop_contact_id"]


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


def fit_regression(df, flag_pct=95, random_state=42):
    X = df[REG_NUMERIC_FEATURES + REG_CATEGORICAL_FEATURES]
    y = df["invoice_amount"]

    pre = ColumnTransformer([
        ("num", StandardScaler(), REG_NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), REG_CATEGORICAL_FEATURES),
    ])
    model = Pipeline([("pre", pre), ("gbm", HistGradientBoostingRegressor(random_state=random_state))])

    oof_pred = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=random_state))
    residual = y.values - oof_pred
    threshold = np.percentile(np.abs(residual), flag_pct)
    flag = np.abs(residual) > threshold

    model.fit(X, y)  # final fit on all data, for scoring new points later

    return {"model": model, "X": X, "y": y, "oof_pred": oof_pred,
            "residual": residual, "threshold": threshold, "flag": flag}
