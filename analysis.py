"""Analysis helpers for the Recruitment-to-Exploitation Pattern Mapping prototype.

Pure functions (no Streamlit) so they can be tested separately from the UI.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_score,
    recall_score, silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
STAGES = ["Recruitment", "Control", "Exploitation", "Profile", "Other"]

# Values that mean "not recorded" in CTDC-style files.
MISSING_TOKENS = {
    "", "-99", "-99.0", "not specified", "unspecified", "unknown", "n/a", "na",
    "none", "null", "nan", "missing", "not known", "don't know",
}

STAGE_KEYWORDS = {
    "Recruitment": ["recruit", "lure", "offer", "promise", "relationship", "advert"],
    "Control": ["control", "means", "debt", "bondage", "threat", "abuse", "withh",
                "restrict", "isolat", "coerc", "violence", "psycholog", "document"],
    "Exploitation": ["exploit", "labour", "labor", "sexual", "forced", "slavery",
                     "organ", "begging", "industry", "sector"],
    "Profile": ["gender", "age", "citizen", "status", "entry", "year", "datasource",
                "source", "country", "nationality"],
}


# ------------------------------------------------------------------ cleaning
def clean_missing(df):
    """Turn codes such as -99 / 'Not specified' / 'Unknown' into real NaN."""
    out = df.copy()
    converted = 0
    for col in out.columns:
        s = out[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            mask = s.isin([-99, -99.0])
        elif pd.api.types.is_bool_dtype(s):
            continue
        else:
            txt = s.astype("string").str.strip().str.lower()
            mask = txt.isin(MISSING_TOKENS).fillna(False)
            out[col] = s.astype(object).where(~mask, np.nan)
            converted += int((mask & s.notna()).sum())
            continue
        converted += int(mask.sum())
        out.loc[mask, col] = np.nan
    return out, converted


def is_id_like(col):
    c = col.lower().strip().replace(" ", "_")
    return c in {"id", "recordid", "record_id", "caseid", "case_id", "index"} or c.endswith("_id")


def is_cluster_col(col):
    c = col.lower()
    return "cluster" in c or "kmeans" in c


def is_source_col(col):
    c = col.lower()
    return c in {"datasource", "dataset", "source", "data_source"}


def is_age_derived(col):
    """Columns that are all views of the same thing: the person's age group.

    e.g. ageBroad (bracket), majorityStatus (Adult/Minor now),
    majorityStatusAtExploit (Adult/Minor at exploitation), majorityEntry.
    ageBroad fully determines majorityStatus, and the status columns constrain one
    another, so using several together (or one as the target and another as a
    predictor) leaks information.
    """
    c = col.lower().replace("_", "").replace(" ", "")
    return any(k in c for k in ("agebroad", "majoritystatus", "majorityentry",
                                "majorityage", "ageatexploit")) or c == "age"


def suggest_stage(col):
    if is_cluster_col(col):
        return "Other"
    if is_age_derived(col):      # 'AtExploit' in the name does NOT make it an exploitation type
        return "Profile"
    c = col.lower()
    for stage in ["Recruitment", "Control", "Exploitation", "Profile"]:
        if any(k in c for k in STAGE_KEYWORDS[stage]):
            return stage
    return "Other"


def usable_columns(df):
    """Columns that are not IDs / earlier cluster labels / empty / constant."""
    cols = []
    for c in df.columns:
        if is_id_like(c) or is_cluster_col(c):
            continue
        if df[c].notna().sum() == 0 or df[c].nunique(dropna=True) <= 1:
            continue
        cols.append(c)
    return cols



# ------------------------------------------------- redundancy / leakage checks
def determines(a, b, min_cov=0.2, purity=0.99, max_levels=30):
    """True if the recorded values of `a` (almost) fully determine `b`.

    Used to find columns that are just a re-coding of another column
    (e.g. ageBroad -> majorityStatus). Evaluated only on rows where both are recorded.
    """
    m = a.notna() & b.notna()
    if m.sum() < 30 or m.mean() < min_cov:
        return False
    a2, b2 = a[m].astype(str), b[m].astype(str)
    na, nb = a2.nunique(), b2.nunique()
    if na < 2 or nb < 2 or na > max_levels or len(a2) / na < 20:
        return False
    if b2.value_counts(normalize=True).iloc[0] > 0.95:   # target almost constant: not informative
        return False
    ct = pd.crosstab(a2, b2)
    return ct.max(axis=1).sum() / ct.values.sum() >= purity


def drop_redundant(work):
    """Remove columns that carry the same information as another selected column.

    1) keep a single age representation (ageBroad if present);
    2) drop any column that is a deterministic re-coding of another selected column
       (the finer / determining column is kept).
    Returns (reduced frame, [(dropped_column, reason), ...]).
    """
    notes, drop = [], set()
    age = [c for c in work.columns if is_age_derived(c)]
    if len(age) > 1:
        keep = next((c for c in age if c.lower() == "agebroad"), age[0])
        for c in age:
            if c != keep:
                drop.add(c)
                notes.append((c, f"same age information as '{keep}'"))
    cols = [c for c in work.columns if c not in drop
            and (not pd.api.types.is_numeric_dtype(work[c]) or work[c].nunique() <= 30)]
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if a in drop or b in drop:
                continue
            if determines(work[a], work[b]):
                drop.add(b)
                notes.append((b, f"fully determined by '{a}'"))
            elif determines(work[b], work[a]):
                drop.add(a)
                notes.append((a, f"fully determined by '{b}'"))
    return work.drop(columns=list(drop)), notes


def leak_screen(X, y, target):
    """Predictors that restate the target and must not be used to predict it."""
    out = {}
    for c in X.columns:
        if is_age_derived(target) and is_age_derived(c):
            out[c] = f"another age-status field; '{target}' and '{c}' describe the same attribute"
        elif determines(X[c], y):
            out[c] = f"values fully determine '{target}'"
        elif determines(y, X[c]):
            out[c] = f"is a re-coding of '{target}'"
    return out

# ----------------------------------------------------------------- matrices
def build_matrix(work, max_col_missing=0.6, max_row_missing=0.5, max_categories=20):
    """Encode mixed data for distance-based clustering.

    * columns that are mostly empty are dropped (and reported)
    * rows that are mostly empty are dropped (and reported)
    * continuous numerics: median-imputed + standardised
    * binary / categorical: one-hot, each block down-weighted by 1/sqrt(levels)
      so a many-level column cannot dominate the distance
    """
    info = {"dropped_cols": {}, "rows_in": len(work)}
    work = work.replace([np.inf, -np.inf], np.nan)

    miss = work.isna().mean()
    for c, m in miss.items():
        if m > max_col_missing:
            info["dropped_cols"][c] = round(float(m), 3)
    work = work.drop(columns=list(info["dropped_cols"]))
    if work.shape[1] < 2:
        raise ValueError(
            "Fewer than two features remain after removing columns with more than "
            f"{max_col_missing:.0%} missing values. Choose better-filled features "
            "or raise the missing-value limit."
        )

    row_miss = work.isna().mean(axis=1)
    keep = row_miss <= max_row_missing
    info["rows_dropped_missing"] = int((~keep).sum())
    work = work.loc[keep]
    if len(work) < 20:
        raise ValueError("Too few complete-enough records remain to cluster.")

    blocks, names = [], []
    for c in work.columns:
        s = work[c]
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if numeric and s.nunique(dropna=True) > 2:
            v = s.fillna(s.median()).to_numpy(float)
            sd = v.std()
            blocks.append(((v - v.mean()) / (sd if sd > 0 else 1.0)).reshape(-1, 1))
            names.append(c)
        else:
            cat = s.astype(object).where(s.notna(), "Missing").astype(str)
            vc = cat.value_counts()
            top = set(vc.index[:max_categories])
            cat = cat.where(cat.isin(top), "Other")
            dummies = pd.get_dummies(cat, prefix=c, dtype=float)
            w = 1.0 / np.sqrt(max(dummies.shape[1], 1))
            blocks.append(dummies.to_numpy() * w * np.sqrt(2))
            names.extend(dummies.columns.tolist())
    X = np.concatenate(blocks, axis=1)
    return X, names, work, info


# ---------------------------------------------------------------- clustering
def k_sweep(X, ks=range(2, 9), sample=2000):
    rows = []
    for k in ks:
        if len(X) <= k:
            break
        km = KMeans(n_clusters=k, random_state=SEED, n_init=5).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=min(sample, len(X)),
                               random_state=SEED)
        rows.append({"K": k, "Silhouette": round(float(sil), 3),
                     "Inertia": round(float(km.inertia_), 1)})
    return pd.DataFrame(rows)


def kmeans_stability(X, k, n_seeds=5):
    """Mean ARI between runs with different random starts (1.0 = fully stable)."""
    runs = [KMeans(n_clusters=k, random_state=s, n_init=1).fit_predict(X)
            for s in range(n_seeds)]
    aris = [adjusted_rand_score(runs[i], runs[j])
            for i in range(n_seeds) for j in range(i + 1, n_seeds)]
    return float(np.mean(aris))


def run_clustering(X, k, hier_max=1500):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=SEED)
    pca = PCA(n_components=2, random_state=SEED)
    pts = pca.fit_transform(X)

    rng = np.random.default_rng(SEED)
    idx = (np.arange(len(X)) if len(X) <= hier_max
           else np.sort(rng.choice(len(X), hier_max, replace=False)))
    hier = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X[idx])
    return {
        "labels": labels, "silhouette": float(sil), "pca_points": pts,
        "pca_var": float(pca.explained_variance_ratio_.sum()),
        "stability": kmeans_stability(X, k),
        "hier_idx": idx, "hier_labels": hier,
        "ari_km_vs_hier": float(adjusted_rand_score(labels[idx], hier)),
    }


def explain_clusters_with_tree(X, labels, names, max_depth=4):
    """Surrogate tree: how well do the features reproduce the K-Means groups?

    The scores are *fidelity to K-Means labels*, not accuracy against real classes.
    """
    strat = labels if pd.Series(labels).value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, labels, test_size=0.2,
                                          random_state=SEED, stratify=strat)
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=5,
                                  random_state=SEED).fit(Xtr, ytr)
    pred = tree.predict(Xte)
    imp = (pd.Series(tree.feature_importances_, index=names)
           .sort_values(ascending=False).head(15).round(3))
    return {
        "tree": tree, "y_test": yte, "pred": pred, "importance": imp,
        "accuracy": accuracy_score(yte, pred),
        "f1_weighted": f1_score(yte, pred, average="weighted", zero_division=0),
    }


def cluster_profile(work, labels, stage_map):
    """Per cluster: most common category (share) or mean, grouped by stage."""
    labels = pd.Series(labels, index=work.index, name="cluster")
    ks = sorted(labels.unique())
    rows = []
    for c in work.columns:
        row = {"Stage": stage_map.get(c, "Other"), "Feature": c}
        for k in ks:
            s = work.loc[labels == k, c]
            numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
            if numeric and work[c].nunique(dropna=True) > 2:
                row[f"Cluster {k}"] = f"mean {s.mean():.1f}" if s.notna().any() else "n/a"
            else:
                filled = s.dropna()
                miss = s.isna().mean()
                if len(filled):
                    top = filled.astype(str).value_counts(normalize=True)
                    row[f"Cluster {k}"] = (f"{top.index[0]} ({top.iloc[0]:.0%} of recorded; "
                                           f"{miss:.0%} missing)")
                else:
                    row[f"Cluster {k}"] = "all missing"
        rows.append(row)
    out = pd.DataFrame(rows)
    out["_o"] = out["Stage"].map({s: i for i, s in enumerate(STAGES)})
    return out.sort_values("_o").drop(columns="_o").reset_index(drop=True)


# -------------------------------------------------------------- association
def cramers_v(x, y):
    """Bias-corrected Cramer's V between two categorical series (NaN = 'Missing')."""
    x = x.astype(object).where(x.notna(), "Missing").astype(str)
    y = y.astype(object).where(y.notna(), "Missing").astype(str)
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    r, k = table.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def association_table(df, target_series, features, top=15):
    rows = []
    for c in features:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 6:
            try:
                s = pd.qcut(s, 5, duplicates="drop").astype(object)
            except ValueError:
                continue
        rows.append({"Feature": c, "Cramér's V": round(cramers_v(s, target_series), 3)})
    return (pd.DataFrame(rows).sort_values("Cramér's V", ascending=False)
            .head(top).reset_index(drop=True))


# --------------------------------------------------------------- supervised
def _one_hot():
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore")


def _preprocessor(X):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    parts = []
    if num:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("sc", StandardScaler())]), num))
    if cat:
        parts.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("oh", _one_hot())]), cat))
    return ColumnTransformer(parts, remainder="drop")


def target_candidates(df, min_rows=50, max_classes=30):
    out = []
    for c in df.columns:
        if is_id_like(c) or is_cluster_col(c):
            continue
        n = df[c].nunique(dropna=True)
        if 2 <= n <= max_classes and df[c].notna().sum() >= min_rows:
            out.append(c)
    return out


def default_prediction_features(df, target):
    """Everything usable except the target, IDs, cluster labels and other
    *exploitation-stage* columns (they usually restate the target = leakage)."""
    return [c for c in usable_columns(df)
            if c != target and suggest_stage(c) != "Exploitation"
            and not is_source_col(c)
            and not (is_age_derived(target) and is_age_derived(c))]


def prepare_supervised(df, target, features, min_class=10, max_col_missing=0.8,
                       max_rows=20000, screen_leakage=True):
    info = {"dropped_features": {}, "dropped_classes": {}, "leak_excluded": {}}
    work = df[features + [target]].copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
    y = work[target].astype(str).str.strip()
    counts = y.value_counts()
    small = counts[counts < min_class]
    info["dropped_classes"] = small.to_dict()
    keep = y.isin(counts[counts >= min_class].index)
    work, y = work.loc[keep], y.loc[keep]
    if y.nunique() < 2:
        raise ValueError(f"Fewer than two classes have at least {min_class} records.")

    X = work[features].copy()
    for c in list(X.columns):
        s = X[c]
        if s.isna().mean() > max_col_missing:
            info["dropped_features"][c] = "mostly missing"
        elif s.nunique(dropna=True) <= 1:
            info["dropped_features"][c] = "constant"
        elif (not pd.api.types.is_numeric_dtype(s)) and s.nunique(dropna=True) > 100:
            info["dropped_features"][c] = "ID-like / free text (>100 distinct values)"
    X = X.drop(columns=list(info["dropped_features"]))
    if screen_leakage:
        info["leak_excluded"] = leak_screen(X, y, target)
        X = X.drop(columns=list(info["leak_excluded"]))
    if X.shape[1] == 0:
        raise ValueError("No usable predictor columns remain.")
    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype(int)
        elif not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    if len(X) > max_rows:
        X, _, y, _ = train_test_split(X, y, train_size=max_rows,
                                      random_state=SEED, stratify=y)
    info["rows_used"] = len(X)
    return X, y, info


def run_supervised(df, target, features, test_size=0.2, screen_leakage=True):
    X, y, info = prepare_supervised(df, target, features, screen_leakage=screen_leakage)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          random_state=SEED, stratify=y)
    classes = sorted(y.unique())
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=SEED),
    }
    min_class = int(y.value_counts().min())
    n_splits = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    base = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    base_pred = base.predict(Xte)
    results = {
        "Baseline (always majority class)": {
            "Accuracy": accuracy_score(yte, base_pred),
            "Balanced accuracy": balanced_accuracy_score(yte, base_pred),
            "Macro precision": precision_score(yte, base_pred, average="macro", zero_division=0),
            "Macro recall": recall_score(yte, base_pred, average="macro", zero_division=0),
            "Macro F1": f1_score(yte, base_pred, average="macro", zero_division=0),
            f"CV Macro F1 ({n_splits}-fold)": float("nan"),
        }}
    fitted, reports, matrices = {}, {}, {}
    for name, clf in models.items():
        pipe = Pipeline([("prep", _preprocessor(X)), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        cvs = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
        results[name] = {
            "Accuracy": accuracy_score(yte, pred),
            "Balanced accuracy": balanced_accuracy_score(yte, pred),
            "Macro precision": precision_score(yte, pred, average="macro", zero_division=0),
            "Macro recall": recall_score(yte, pred, average="macro", zero_division=0),
            "Macro F1": f1_score(yte, pred, average="macro", zero_division=0),
            f"CV Macro F1 ({n_splits}-fold)": float(cvs.mean()),
        }
        fitted[name] = pipe
        reports[name] = pd.DataFrame(classification_report(
            yte, pred, labels=classes, output_dict=True, zero_division=0)).T
        matrices[name] = confusion_matrix(yte, pred, labels=classes)

    summary = pd.DataFrame(results).T.round(3)
    best = summary.drop(index="Baseline (always majority class)")["Macro F1"].idxmax()

    n_imp = min(1500, len(Xte))
    Xi = Xte.sample(n=n_imp, random_state=SEED) if n_imp < len(Xte) else Xte
    yi = yte.loc[Xi.index]
    perm = permutation_importance(fitted[best], Xi, yi, scoring="f1_macro",
                                  n_repeats=5, random_state=SEED, n_jobs=1)
    importance = (pd.DataFrame({"Feature": X.columns,
                                "Importance (drop in macro-F1)": perm.importances_mean.round(4),
                                "Std": perm.importances_std.round(4)})
                  .sort_values("Importance (drop in macro-F1)", ascending=False)
                  .reset_index(drop=True))

    assoc = association_table(pd.concat([X, y.rename(target)], axis=1), y, list(X.columns))
    return {
        "info": info, "classes": classes, "summary": summary, "best": best,
        "reports": reports, "matrices": matrices, "importance": importance,
        "association": assoc, "class_counts": y.value_counts(),
        "majority_share": float(y.value_counts(normalize=True).iloc[0]),
        "target": target, "features": list(X.columns), "n_test": len(Xte),
    }
