Recruitment-to-Exploitation Pattern Mapping - Prototype v2

Files
- app.py            Streamlit interface
- analysis.py       All ML / cleaning logic (kept separate so it can be tested)
- requirements.txt  Packages
- sample_synthetic_data.csv  SYNTHETIC demo data (random, not real cases) for trying the app

Run (Windows)
1. Extract the ZIP and open the folder in VS Code.
2. python -m pip install -r requirements.txt
3. python -m streamlit run app.py
4. Upload the permitted CTDC CSV/Excel file (or the synthetic sample) in the sidebar.

What the app does
- Cleaning: converts codes such as -99 / "Not specified" / "Unknown" to real missing
  values; optional duplicate removal.
- Clustering Analysis (unsupervised): K-Means and Hierarchical (Ward) clustering on
  features you assign to Recruitment / Control / Exploitation / Profile stages.
  Columns that are mostly empty are dropped; one-hot blocks are down-weighted so
  many-level columns do not dominate. Reports silhouette, restart stability, K-Means vs
  Hierarchical agreement (ARI), a K-selection sweep, PCA map, cluster profiles by stage,
  and optional comparison of clusters with a recorded column.
  A surrogate Decision Tree shows which variables separate the clusters; its scores are
  agreement with K-Means labels, NOT accuracy against real categories.
- Exploitation Category Prediction (supervised): Logistic Regression, Random Forest and
  Decision Tree predict a RECORDED category (e.g. type of exploitation) with a stratified
  split, 5-fold CV, a majority-class baseline, macro-F1 / balanced accuracy, per-class
  report and confusion matrix. Key factors: permutation importance and Cramer's V.
  Other exploitation-stage columns are excluded from predictors by default to avoid
  leakage, and near-perfect scores trigger a leakage warning.

Leakage / redundancy checks (added)
- ageBroad, majorityStatus, majorityStatusAtExploit and majorityEntry all describe the
  person's age group. ageBroad fully determines majorityStatus, and the status columns
  constrain one another. They are treated as ONE attribute:
  * Clustering: only one age field is kept (ageBroad if present); any column that is a
    deterministic re-coding of another selected column is dropped (option can be switched off).
  * Prediction: if the target is an age-status field, all other age fields are excluded
    automatically, and any predictor that fully determines (or re-codes) the target is
    excluded and reported on screen.
  * majorityStatusAtExploit is classed as a Profile (age) field, NOT an exploitation type.
- Check on the provided data: predicting majorityStatus with ageBroad included scored
  macro-F1 0.987; with the leaking columns removed it falls to 0.59. Predicting
  majorityStatusAtExploit: 0.90 -> 0.72.

Important limits
- The dataset must contain recruitment / control / exploitation columns. A file with only
  demographics (gender, age, citizenship, ...) cannot show recruitment-to-exploitation
  patterns; the app warns when this is the case.
- Results are associations in recorded data, not causes, and are not real-world detection
  performance. Do not use them to make decisions about individuals.
- Clusters are exploratory groups, not confirmed trafficking categories.
