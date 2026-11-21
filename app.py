import warnings
warnings.filterwarnings("ignore")

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import plot_tree
from scipy.cluster.hierarchy import dendrogram, linkage

import analysis as an

st.set_page_config(page_title="Recruitment-to-Exploitation Pattern Mapping",
                   page_icon="🔍", layout="wide")
st.title("🔍 Recruitment-to-Exploitation Pattern Mapping")
st.write("Machine-learning analysis of recurring recruitment, control and exploitation patterns")
st.info(
    "Research prototype for recorded or synthetic case data. It groups case patterns "
    "and studies which recorded factors go with recorded exploitation categories. "
    "It does not identify or label individual victims or suspects, and its output "
    "must not be used to make decisions about individuals."
)


# ------------------------------------------------------------------ loading
@st.cache_data(show_spinner=False)
def load_dataset(file_bytes, filename):
    if filename.lower().endswith(".xlsx"):
        return pd.read_excel(BytesIO(file_bytes))
    return pd.read_csv(BytesIO(file_bytes))


st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Module", [
    "Home", "Dataset Analysis", "Data Preprocessing", "Pattern Visualization",
    "Clustering Analysis", "Exploitation Category Prediction",
])
uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset (CSV or Excel)", type=["csv", "xlsx"],
    help="Excel files should have the data on the first worksheet.")
opt_missing = st.sidebar.checkbox(
    "Treat codes like -99 / 'Not specified' / 'Unknown' as missing", value=True)
opt_dups = st.sidebar.checkbox("Remove exact duplicate rows", value=True)

raw_df, df, n_converted, n_dups = None, None, 0, 0
if uploaded_file is not None:
    try:
        raw_df = load_dataset(uploaded_file.getvalue(), uploaded_file.name)
        if raw_df.empty:
            st.error("The uploaded file contains no data rows.")
            raw_df = None
        else:
            df = raw_df.copy()
            if opt_missing:
                df, n_converted = an.clean_missing(df)
            if opt_dups:
                before = len(df)
                df = df.drop_duplicates().reset_index(drop=True)
                n_dups = before - len(df)
            st.sidebar.success(f"Loaded {len(df):,} records")
    except Exception as error:
        st.sidebar.error(f"Unable to read the dataset: {error}")


def need_data():
    if df is None:
        st.warning("Please upload your CSV or Excel dataset using the sidebar.")
        return True
    return False


# --------------------------------------------------------------------- home
if page == "Home":
    st.header("Welcome to the Research Prototype")
    c1, c2, c3 = st.columns(3)
    c1.metric("Unsupervised", "K-Means + Hierarchical")
    c2.metric("Supervised", "LogReg + RF + Tree")
    c3.metric("Files", "CSV / Excel")
    st.markdown(
        """
**Modules**
1. **Dataset Analysis / Preprocessing / Pattern Visualization** – inspect and clean the data
2. **Clustering Analysis** – group similar case patterns using features you assign to the
   *Recruitment, Control, Exploitation* and *Profile* stages; K selection, stability checks,
   PCA map, cluster profiles by stage
3. **Exploitation Category Prediction** – predict a *recorded* exploitation category with three
   models against a majority-class baseline, and rank the key factors (permutation importance
   and Cramér's V)
        """
    )
    st.warning(
        "Results depend on the dataset, class balance and chosen features. They describe "
        "associations in recorded data; they are not proof that a person or advertisement "
        "is connected to trafficking. The dataset must contain recruitment / control / "
        "exploitation columns for the stage analysis to be meaningful."
    )
    if df is None:
        st.caption("Upload your dataset using the left sidebar to begin.")


# --------------------------------------------------------- dataset analysis
elif page == "Dataset Analysis":
    st.header("Dataset Analysis")
    if not need_data():
        c1, c2, c3 = st.columns(3)
        c1.metric("Records (after cleaning)", f"{len(df):,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        st.caption(f"{n_converted:,} cells converted from missing-value codes; "
                   f"{n_dups:,} duplicate rows removed.")
        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)
        st.subheader("Columns")
        info = pd.DataFrame({
            "Column": df.columns.astype(str),
            "Type": df.dtypes.astype(str).values,
            "Missing %": (df.isna().mean() * 100).round(1).values,
            "Unique": df.nunique(dropna=True).values,
            "Suggested stage": [an.suggest_stage(c) for c in df.columns],
        })
        st.dataframe(info, use_container_width=True)
        st.subheader("Statistical summary")
        st.dataframe(df.describe(include="all").T, use_container_width=True)


# ------------------------------------------------------------ preprocessing
elif page == "Data Preprocessing":
    st.header("Data Preprocessing")
    if not need_data():
        st.write("Original shape:", raw_df.shape, "→ cleaned shape:", df.shape)
        st.write(f"Cells converted to missing (-99 / Unknown / Not specified …): {n_converted:,}")
        st.write(f"Exact duplicate rows removed: {n_dups:,}")
        st.subheader("Missing values per column (after cleaning)")
        miss = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
        st.bar_chart(miss)
        high = miss[miss > 60]
        if len(high):
            st.warning("Columns with more than 60% missing values are dropped "
                       "from clustering by default: " + ", ".join(high.index))
        st.download_button("Download cleaned dataset",
                           df.to_csv(index=False).encode("utf-8"),
                           "cleaned_dataset.csv", "text/csv")


# ------------------------------------------------------------ visualization
elif page == "Pattern Visualization":
    st.header("Pattern Visualization")
    if not need_data():
        cats = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])
                or df[c].nunique() <= 12]
        if cats:
            col = st.selectbox("Category to visualize", cats)
            st.bar_chart(df[col].fillna("Missing").astype(str).value_counts().head(15))
            other = st.selectbox("Cross-tabulate with (optional)", ["(none)"] + cats)
            if other != "(none)" and other != col:
                ct = pd.crosstab(df[col].fillna("Missing").astype(str),
                                 df[other].fillna("Missing").astype(str), normalize="index")
                st.caption(f"Share of each {other} value within each {col} value")
                st.dataframe((ct * 100).round(1), use_container_width=True)
        nums = df.select_dtypes(include=np.number).columns.tolist()
        if nums:
            st.subheader("Numerical columns")
            n = st.selectbox("Numerical column", nums)
            st.bar_chart(df[n].dropna().value_counts().head(15).sort_index())


# --------------------------------------------------------------- clustering
elif page == "Clustering Analysis":
    st.header("Clustering and Pattern Mapping")
    st.write("Group similar case patterns with K-Means and Hierarchical (Ward) clustering, "
             "map them with PCA, and read each cluster as a recruitment → control → "
             "exploitation profile.")
    if not need_data():
        st.info("Clusters are exploratory groups of records with similar recorded values. "
                "They are not confirmed trafficking categories and say nothing about "
                "any individual.")
        cands = an.usable_columns(df)
        if len(cands) < 2:
            st.error("The dataset needs at least two usable columns.")
        else:
            suggested = {c: an.suggest_stage(c) for c in cands}
            rce = [c for c in cands if suggested[c] in ("Recruitment", "Control", "Exploitation")]
            defaults = {s: [c for c in cands if suggested[c] == s]
                        for s in ("Recruitment", "Control", "Exploitation")}
            defaults["Profile"] = []
            defaults["Other"] = []
            if len(rce) < 2:   # e.g. demographics-only data
                defaults["Profile"] = [c for c in cands if suggested[c] == "Profile"
                                       and not an.is_source_col(c)]
                defaults["Other"] = [c for c in cands if suggested[c] == "Other"]
                st.warning(
                    "No recruitment, control or exploitation columns were recognised in this "
                    "dataset, so the clusters can only describe demographic/profile "
                    "patterns, not recruitment-to-exploitation patterns. Add such columns "
                    "(for example recruiter relationship, means of control, type of "
                    "exploitation) to the file, or assign columns manually below.")

            st.subheader("1. Assign features to stages")
            chosen, stage_map = [], {}
            for stage in an.STAGES:
                if stage == "Exploitation":
                    help_txt = ("Include only if you want exploitation type to shape the "
                                "clusters; leave empty to compare clusters with it afterwards.")
                else:
                    help_txt = None
                picks = st.multiselect(f"{stage} features", cands,
                                       default=defaults[stage], key=f"stage_{stage}",
                                       help=help_txt)
                for c in picks:
                    if c not in stage_map:
                        stage_map[c] = stage
                        chosen.append(c)
            if any(an.is_source_col(c) for c in chosen):
                st.warning("A data-source column is selected: clusters will probably "
                           "just separate sources (e.g. hotline vs case management).")

            c1, c2 = st.columns(2)
            k_value = c1.slider("2. Number of clusters (K)", 2, 8, 3)
            max_miss = c2.slider("Drop columns with more than this share missing",
                                 0.2, 0.95, 0.6, 0.05)
            do_sweep = st.checkbox("Show K selection (silhouette / elbow)", value=True)
            dedupe = st.checkbox(
                "Remove redundant columns (keep one age field; drop columns that are a "
                "re-coding of another)", value=True,
                help="ageBroad, majorityStatus, majorityStatusAtExploit and majorityEntry all "
                     "describe age group. Keeping several counts age several times and "
                     "forms clusters that are just 'minor vs adult'.")
            if len(chosen) >= 2:
                _, preview = an.drop_redundant(df[chosen].head(5000))
                if preview and dedupe:
                    st.info("Will be removed as redundant: " + "; ".join(
                        f"{c} ({why})" for c, why in preview))
                elif preview:
                    st.warning("Redundant columns are kept, so the same information is "
                               "counted more than once: " + "; ".join(
                                   f"{c} ({why})" for c, why in preview))

            if st.button("Run Clustering and Analysis", type="primary"):
                if len(chosen) < 2:
                    st.error("Select at least two features.")
                else:
                    try:
                        with st.spinner("Clustering..."):
                            work = df[chosen].copy()
                            removed = []
                            if dedupe:
                                work, removed = an.drop_redundant(work)
                            if len(work) > 5000:
                                work = work.sample(n=5000, random_state=an.SEED)
                            X, names, work2, info = an.build_matrix(work, max_col_missing=max_miss)
                            if len(X) <= k_value:
                                raise ValueError("There must be more records than clusters.")
                            res = an.run_clustering(X, k_value)
                            res["sweep"] = an.k_sweep(X) if do_sweep else None
                            res["surrogate"] = an.explain_clusters_with_tree(X, res["labels"], names)
                            res["X"], res["names"], res["work"], res["info"] = X, names, work2, info
                            res["k"] = k_value
                            res["removed"] = removed
                            res["stage_map"] = {c: stage_map[c] for c in work2.columns}
                        st.session_state["cl"] = res
                    except Exception as error:
                        st.session_state.pop("cl", None)
                        st.error(f"Clustering could not be completed: {error}")

            res = st.session_state.get("cl")
            if res is not None and res["work"].columns.isin(df.columns).all():
                k, labels, work2, info = res["k"], res["labels"], res["work"], res["info"]
                st.success(f"Clustered {len(labels):,} records with {len(res['names'])} encoded variables.")
                if res.get("removed"):
                    st.caption("Redundant columns removed: " + "; ".join(
                        f"{c} ({why})" for c, why in res["removed"]))
                if info["dropped_cols"]:
                    st.warning("Dropped for too many missing values: " + ", ".join(
                        f"{c} ({m:.0%})" for c, m in info["dropped_cols"].items()))
                if info["rows_dropped_missing"]:
                    st.caption(f"{info['rows_dropped_missing']:,} records with most selected "
                               "fields empty were excluded.")

                st.subheader("Cluster sizes")
                sizes = pd.Series(labels).value_counts().sort_index()
                sizes.index = [f"Cluster {i}" for i in sizes.index]
                st.bar_chart(sizes)

                m1, m2, m3 = st.columns(3)
                m1.metric("Silhouette", f"{res['silhouette']:.3f}")
                m2.metric("Stability (ARI, 5 restarts)", f"{res['stability']:.2f}")
                m3.metric("K-Means vs Hierarchical (ARI)", f"{res['ari_km_vs_hier']:.2f}")
                st.caption("Silhouette above ~0.25 suggests some structure, below ~0.15 weak. "
                           "Stability near 1 means the clusters don't depend on the random "
                           "start. ARI between the two methods near 1 means they find the "
                           "same groups (computed on the same records).")

                if res["sweep"] is not None and len(res["sweep"]):
                    st.subheader("Choosing K")
                    sw = res["sweep"].set_index("K")
                    a, b = st.columns(2)
                    a.caption("Silhouette (higher is better)")
                    a.line_chart(sw["Silhouette"])
                    b.caption("Inertia (look for an elbow)")
                    b.line_chart(sw["Inertia"])
                    st.caption("Silhouette tends to rise with K on one-hot data, so prefer a K "
                               "where the clusters are also interpretable and stable.")

                st.subheader("Cluster profiles by stage")
                prof = an.cluster_profile(work2, labels, res["stage_map"])
                st.dataframe(prof, use_container_width=True)

                st.subheader("PCA map")
                st.scatter_chart(pd.DataFrame({
                    "Principal Component 1": res["pca_points"][:, 0],
                    "Principal Component 2": res["pca_points"][:, 1],
                    "Cluster": [f"Cluster {i}" for i in labels]}),
                    x="Principal Component 1", y="Principal Component 2", color="Cluster")
                st.caption(f"The 2 components keep {res['pca_var']*100:.1f}% of the variance; "
                           "overlap in this plot does not mean the clusters are meaningless.")

                st.subheader("Hierarchical clustering (Ward)")
                hX = res["X"][res["hier_idx"]]
                fig, ax = plt.subplots(figsize=(12, 4.5))
                dendrogram(linkage(hX, method="ward"), truncate_mode="lastp", p=30,
                           leaf_rotation=45, leaf_font_size=8, ax=ax)
                ax.set_xlabel("Merged record groups")
                ax.set_ylabel("Distance")
                st.pyplot(fig)
                plt.close(fig)
                hs = pd.Series(res["hier_labels"]).value_counts().sort_index()
                hs.index = [f"Cluster {i}" for i in hs.index]
                st.caption(f"Based on {len(hX):,} records.")
                st.bar_chart(hs)

                st.subheader("Compare clusters with a recorded column")
                compare_opts = ["(none)"] + [c for c in df.columns if not an.is_cluster_col(c)
                                             and 2 <= df[c].nunique(dropna=True) <= 30]
                comp = st.selectbox("Recorded column (not used unless you selected it above)",
                                    compare_opts)
                if comp != "(none)":
                    vals = df.loc[work2.index, comp] if work2.index.isin(df.index).all() else None
                    if vals is not None:
                        v = vals.astype(object).where(vals.notna(), "Missing").astype(str)
                        ct = pd.crosstab(pd.Series(labels, index=work2.index,
                                                   name="Cluster").map(lambda i: f"Cluster {i}"),
                                         v, normalize="index")
                        st.dataframe((ct * 100).round(1), use_container_width=True)
                        cv = an.cramers_v(v, pd.Series(labels, index=v.index))
                        st.caption(f"Association between clusters and {comp}: "
                                   f"Cramér's V = {cv:.2f} (0 none, 1 perfect).")

                st.subheader("Decision Tree explaining the K-Means clusters")
                sg = res["surrogate"]
                st.caption("Surrogate tree: it shows which variables separate the clusters. "
                           "Its scores measure agreement with the K-Means labels, which were "
                           "built from the same variables, so they will be high by design "
                           "and are NOT accuracy against real exploitation categories.")
                a, b = st.columns(2)
                a.metric("Agreement with K-Means (accuracy)", f"{sg['accuracy']:.3f}")
                b.metric("Agreement (weighted F1)", f"{sg['f1_weighted']:.3f}")
                st.write("Most influential variables")
                st.dataframe(sg["importance"].rename("Importance").to_frame(),
                             use_container_width=True)
                fig, ax = plt.subplots(figsize=(18, 8))
                plot_tree(sg["tree"], filled=True, rounded=True, max_depth=3, fontsize=7,
                          feature_names=res["names"],
                          class_names=[f"Cluster {i}" for i in range(k)], ax=ax)
                st.pyplot(fig)
                plt.close(fig)

                exp = work2.copy()
                exp["Cluster"] = labels
                st.download_button("Download analysed records with cluster labels",
                                   exp.to_csv(index=False).encode("utf-8"),
                                   "records_with_clusters.csv", "text/csv")
                st.caption("Cluster columns are ignored automatically if you upload this "
                           "file again.")


# --------------------------------------------------------------- prediction
elif page == "Exploitation Category Prediction":
    st.header("Exploitation Category Prediction")
    st.write("Predict a **recorded** category (for example type of exploitation) from "
             "recruitment, control and profile columns, and rank the key factors. "
             "Three models are compared against a majority-class baseline.")
    if not need_data():
        targets = an.target_candidates(df)
        if not targets:
            st.error("No suitable categorical target column was found "
                     "(needs 2–30 classes and at least 50 labelled rows).")
        else:
            pref = [t for t in targets if an.suggest_stage(t) == "Exploitation"]
            order = pref + [t for t in targets if t not in pref]
            target = st.selectbox("Recorded category to predict", order)
            if an.is_age_derived(target):
                st.warning(f"'{target}' is an age-status field, not an exploitation category. "
                           "Other age fields (ageBroad, majorityStatus, ...) restate it and are "
                           "excluded automatically. Choose a column such as type of "
                           "exploitation for the real research question.")
            default_feats = an.default_prediction_features(df, target)
            options = [c for c in an.usable_columns(df) if c != target]
            feats = st.multiselect(
                "Predictor columns", options, default=default_feats, key=f"pf_{target}",
                help="Other exploitation-stage columns are excluded by default because "
                     "they usually restate the target (leakage).")
            leaky = [c for c in feats if an.suggest_stage(c) == "Exploitation"
                     or (an.is_age_derived(target) and an.is_age_derived(c))]
            if leaky:
                st.warning("These predictors look like exploitation outcomes and may leak the "
                           "answer: " + ", ".join(leaky))
            test_size = st.slider("Test share", 0.1, 0.4, 0.2, 0.05)

            if st.button("Train and evaluate models", type="primary"):
                if not feats:
                    st.error("Select at least one predictor column.")
                else:
                    try:
                        with st.spinner("Training three models and computing factor importance..."):
                            st.session_state["sup"] = an.run_supervised(df, target, feats, test_size)
                    except Exception as error:
                        st.session_state.pop("sup", None)
                        st.error(f"Prediction could not be completed: {error}")

            res = st.session_state.get("sup")
            if res is not None and res["target"] in df.columns:
                info = res["info"]
                st.success(f"Target: {res['target']} | {info['rows_used']:,} records | "
                           f"{len(res['classes'])} classes | {len(res['features'])} predictors")
                if info["dropped_classes"]:
                    st.warning("Classes with fewer than 10 records were removed: " + ", ".join(
                        f"{k} ({v})" for k, v in info["dropped_classes"].items()))
                if info["leak_excluded"]:
                    st.error("Excluded to prevent data leakage: " + "; ".join(
                        f"{c} ({why})" for c, why in info["leak_excluded"].items()))
                if info["dropped_features"]:
                    st.caption("Predictors dropped: " + ", ".join(
                        f"{k} ({v})" for k, v in info["dropped_features"].items()))

                st.subheader("Class balance")
                st.bar_chart(res["class_counts"])
                st.caption(f"The majority class is {res['majority_share']:.0%} of records, so "
                           "plain accuracy can look good without learning anything. "
                           "Compare with the baseline and focus on balanced accuracy / macro-F1.")

                st.subheader("Model comparison (held-out test set)")
                st.dataframe(res["summary"], use_container_width=True)
                best = res["best"]
                bf1 = res["summary"].loc[best, "Macro F1"]
                st.write(f"Best by macro-F1: **{best}** ({bf1:.3f})")
                if bf1 > 0.97:
                    st.error("Near-perfect scores usually mean a predictor leaks the target "
                             "(a column that restates it). Check the predictor list.")
                elif bf1 <= res["summary"].iloc[0]["Macro F1"] + 0.02:
                    st.warning("The best model barely beats the majority-class baseline: the "
                               "chosen columns carry little signal for this category.")

                st.subheader("Key factors")
                a, b = st.columns(2)
                a.caption(f"Permutation importance ({best})")
                a.dataframe(res["importance"], use_container_width=True)
                b.caption("Association with the category (Cramér's V)")
                b.dataframe(res["association"], use_container_width=True)
                st.caption("Importance shows what the model relies on; Cramér's V shows "
                           "one-to-one association. Both describe patterns in the recorded "
                           "data, not causes.")

                st.subheader("Confusion matrix and per-class results")
                model_name = st.selectbox("Model", list(res["matrices"].keys()),
                                          index=list(res["matrices"].keys()).index(best))
                cm = res["matrices"][model_name]
                fig, ax = plt.subplots(figsize=(7, 5.5))
                ConfusionMatrixDisplay(cm, display_labels=res["classes"]).plot(
                    ax=ax, colorbar=False, values_format="d", xticks_rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.dataframe(res["reports"][model_name].round(3), use_container_width=True)
                st.caption("These scores are measured against recorded labels in the dataset. "
                           "They describe how well recorded factors predict the recorded "
                           "category here, not real-world detection performance.")


st.sidebar.markdown("---")
st.sidebar.caption("Human Trafficking Research Prototype | ML Project")
