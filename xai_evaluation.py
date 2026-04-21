"""
Explainable AI Evaluation Pipeline
Author: Prachi Puthran | Roll: 23AD1105
Research Paper: Evaluating Explainable AI Methods in High-Stakes ML Systems

Evaluates SHAP and LIME across:
  - Breast Cancer Wisconsin (healthcare)
  - Heart Disease UCI (healthcare)
  - German Credit (finance)

Metrics: Accuracy, F1, Consistency (std dev), Stability (perturbation), Rank Correlation
"""

# ─────────────────────────────────────────────
# 0. INSTALL DEPENDENCIES (run once)
# pip install shap lime scikit-learn pandas numpy matplotlib seaborn scipy ucimlrepo
# ─────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

import shap
import lime
import lime.lime_tabular

import os, json, time

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RANDOM_STATE   = 42
LIME_RUNS      = 20          # repetitions to measure LIME consistency
PERTURB_NOISE  = 0.05        # Gaussian noise std for stability test
N_EXPLAIN      = 30          # instances to explain (keep runtime reasonable)
OUTPUT_DIR     = "xai_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────
# 1. DATASET LOADERS
# ─────────────────────────────────────────────

def load_breast_cancer_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y, "Breast Cancer Wisconsin"


def load_heart_disease_data():
    """UCI Heart Disease (Cleveland). Falls back to synthetic if network unavailable."""
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=45)
        X = ds.data.features.copy()
        y = ds.data.targets.copy().squeeze()
        y = (y > 0).astype(int)          # binary: disease vs no-disease
        X = X.apply(pd.to_numeric, errors="coerce").dropna()
        y = y.loc[X.index].reset_index(drop=True)
        X = X.reset_index(drop=True)
    except Exception:
        print("  [Heart Disease] ucimlrepo unavailable – generating synthetic data")
        rng = np.random.default_rng(RANDOM_STATE)
        n = 303
        X = pd.DataFrame({
            "age":      rng.integers(29, 77, n).astype(float),
            "sex":      rng.integers(0, 2, n).astype(float),
            "cp":       rng.integers(0, 4, n).astype(float),
            "trestbps": rng.integers(94, 200, n).astype(float),
            "chol":     rng.integers(126, 564, n).astype(float),
            "fbs":      rng.integers(0, 2, n).astype(float),
            "restecg":  rng.integers(0, 3, n).astype(float),
            "thalach":  rng.integers(71, 202, n).astype(float),
            "exang":    rng.integers(0, 2, n).astype(float),
            "oldpeak":  rng.uniform(0, 6.2, n),
            "slope":    rng.integers(0, 3, n).astype(float),
            "ca":       rng.integers(0, 4, n).astype(float),
            "thal":     rng.integers(0, 4, n).astype(float),
        })
        y = pd.Series(rng.integers(0, 2, n), name="target")
    return X, y, "Heart Disease (UCI)"


def load_german_credit_data():
    """UCI German Credit. Falls back to synthetic if network unavailable."""
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=144)
        X = ds.data.features.copy()
        y = ds.data.targets.copy().squeeze()
        # encode categoricals
        X = pd.get_dummies(X).astype(float)
        y = (y == 1).astype(int).reset_index(drop=True)
        X = X.reset_index(drop=True)
    except Exception:
        print("  [German Credit] ucimlrepo unavailable – generating synthetic data")
        rng = np.random.default_rng(RANDOM_STATE)
        n = 1000
        X = pd.DataFrame({
            "duration":       rng.integers(4, 72, n).astype(float),
            "credit_amount":  rng.integers(250, 18424, n).astype(float),
            "installment_rate": rng.integers(1, 5, n).astype(float),
            "age":            rng.integers(19, 75, n).astype(float),
            "existing_credits": rng.integers(1, 5, n).astype(float),
            "num_dependents": rng.integers(1, 3, n).astype(float),
        })
        y = pd.Series(rng.integers(0, 2, n), name="target")
    return X, y, "German Credit"


DATASETS = [load_breast_cancer_data, load_heart_disease_data, load_german_credit_data]

# ─────────────────────────────────────────────
# 2. MODEL FACTORY
# ─────────────────────────────────────────────

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Neural Network":      MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=RANDOM_STATE),
    }

# ─────────────────────────────────────────────
# 3. XAI METRICS
# ─────────────────────────────────────────────

def shap_feature_importances(model, X_train, X_test_sample, model_name):
    """Return mean |SHAP| per feature."""
    X_train_arr  = X_train.values if hasattr(X_train, "values") else X_train
    X_test_arr   = X_test_sample.values if hasattr(X_test_sample, "values") else X_test_sample

    if model_name == "Random Forest":
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X_test_arr)
        if isinstance(vals, list):          # multi-class output
            vals = vals[1]
    else:
        background = shap.kmeans(X_train_arr, 10)
        explainer  = shap.KernelExplainer(model.predict_proba, background)
        vals = explainer.shap_values(X_test_arr, nsamples=100)
        if isinstance(vals, list):
            vals = vals[1]

    return np.abs(vals).mean(axis=0)        # shape: (n_features,)


def lime_feature_importances_single_run(explainer, model, instance, n_features):
    """One LIME run for a single instance → feature weight vector."""
    exp = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=n_features,
        num_samples=500,
    )
    weights = dict(exp.as_list())
    # map back to all features by index
    feature_vec = np.zeros(n_features)
    for feat_name, weight in weights.items():
        # LIME names features as "feat_name <= value" etc.; parse index
        for idx in range(n_features):
            if str(idx) in feat_name or feat_name.startswith(str(idx)):
                feature_vec[idx] = abs(weight)
                break
    return feature_vec


def compute_lime_consistency(model, X_train, X_test_sample, feature_names, runs=LIME_RUNS):
    """
    Run LIME `runs` times on each test instance.
    Consistency = 1 - mean(std of feature importance across runs).
    Also returns per-instance std arrays for further analysis.
    """
    X_train_arr = X_train.values if hasattr(X_train, "values") else X_train
    X_test_arr  = X_test_sample.values if hasattr(X_test_sample, "values") else X_test_sample
    n_features  = X_train_arr.shape[1]

    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        X_train_arr,
        feature_names=feature_names,
        class_names=["0", "1"],
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )

    all_stds   = []   # std across runs, per instance
    mean_importances = []

    for inst in X_test_arr:
        run_results = []
        for _ in range(runs):
            exp = lime_exp.explain_instance(
                inst,
                model.predict_proba,
                num_features=n_features,
                num_samples=500,
            )
            raw = {int(k): abs(v) for k, v in exp.local_exp[1]}
            vec = np.array([raw.get(i, 0.0) for i in range(n_features)])
            run_results.append(vec)

        run_results = np.array(run_results)          # (runs, n_features)
        all_stds.append(run_results.std(axis=0))
        mean_importances.append(run_results.mean(axis=0))

    all_stds_arr = np.array(all_stds)                # (n_instances, n_features)
    mean_std     = all_stds_arr.mean()               # scalar – lower = more consistent
    consistency_score = 1 - min(mean_std, 1.0)      # normalise to [0,1]

    return consistency_score, mean_std, np.mean(mean_importances, axis=0)


def compute_shap_stability(model, X_test_sample, shap_vals_original, model_name, X_train, noise=PERTURB_NOISE):
    """
    Perturb test instances with Gaussian noise, recompute SHAP.
    Stability = 1 - mean absolute difference in feature importance rankings.
    """
    X_arr      = X_test_sample.values if hasattr(X_test_sample, "values") else X_test_sample
    X_perturbed = X_arr + np.random.normal(0, noise, X_arr.shape)

    shap_perturbed = shap_feature_importances(model, X_train, pd.DataFrame(X_perturbed, columns=X_test_sample.columns), model_name)

    rank_orig  = np.argsort(-shap_vals_original)
    rank_pert  = np.argsort(-shap_perturbed)

    # normalised rank displacement
    n = len(rank_orig)
    rank_diff  = np.abs(rank_orig - rank_pert).mean() / n
    stability  = 1 - rank_diff
    return float(stability)


def compute_lime_stability(model, X_train, X_test_sample, feature_names, noise=PERTURB_NOISE):
    """Perturb test instances, recompute LIME (1 run each). Stability via rank displacement."""
    X_train_arr = X_train.values if hasattr(X_train, "values") else X_train
    X_test_arr  = X_test_sample.values if hasattr(X_test_sample, "values") else X_test_sample
    n_features  = X_train_arr.shape[1]

    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        X_train_arr,
        feature_names=feature_names,
        class_names=["0", "1"],
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )

    displacements = []
    for inst in X_test_arr:
        def get_vec(instance):
            exp = lime_exp.explain_instance(instance, model.predict_proba,
                                             num_features=n_features, num_samples=500)
            raw = {int(k): abs(v) for k, v in exp.local_exp[1]}
            return np.array([raw.get(i, 0.0) for i in range(n_features)])

        v_orig  = get_vec(inst)
        v_pert  = get_vec(inst + np.random.normal(0, noise, inst.shape))

        r_orig  = np.argsort(-v_orig)
        r_pert  = np.argsort(-v_pert)
        displacements.append(np.abs(r_orig - r_pert).mean() / n_features)

    stability = 1 - np.mean(displacements)
    return float(stability)


def compute_rank_correlation(shap_vals, lime_vals, lr_importances):
    """Spearman rank correlation of SHAP/LIME feature rankings vs LR coefficients (ground truth proxy)."""
    shap_vals_flat = np.array(shap_vals).flatten()
    lime_vals_flat = np.array(lime_vals).flatten()
    lr_flat        = np.array(lr_importances).flatten()

    # Match lengths in case of shape mismatch
    min_len = min(len(shap_vals_flat), len(lime_vals_flat), len(lr_flat))
    shap_vals_flat = shap_vals_flat[:min_len]
    lime_vals_flat = lime_vals_flat[:min_len]
    lr_flat        = lr_flat[:min_len]

    result_shap = spearmanr(shap_vals_flat, lr_flat)
    result_lime = spearmanr(lime_vals_flat, lr_flat)

    rho_shap = float(result_shap.statistic)
    p_shap   = float(result_shap.pvalue)
    rho_lime = float(result_lime.statistic)
    p_lime   = float(result_lime.pvalue)

    return rho_shap, p_shap, rho_lime, p_lime

# ─────────────────────────────────────────────
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────

all_results = []   # list of dicts, one per (dataset × model)

for loader in DATASETS:
    X, y, dataset_name = loader()
    feature_names = list(X.columns)
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}  |  shape: {X.shape}")
    print(f"{'='*60}")

    # Split + scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler  = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=feature_names)

    # Small sample for explanation (speed)
    idx_sample = np.random.choice(len(X_test_sc), size=min(N_EXPLAIN, len(X_test_sc)), replace=False)
    X_explain  = X_test_sc.iloc[idx_sample].reset_index(drop=True)

    # LR importances (ground truth proxy)
    lr_baseline = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_baseline.fit(X_train_sc, y_train)
    lr_importances = np.abs(lr_baseline.coef_[0])

    models = get_models()

    for model_name, model in models.items():
        print(f"\n  ── Model: {model_name}")
        t0 = time.time()

        # Train
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="weighted")
        print(f"     Accuracy: {acc:.4f}  |  F1: {f1:.4f}")

        # ── SHAP ──────────────────────────────────
        print("     Computing SHAP...", end=" ", flush=True)
        shap_vals = shap_feature_importances(model, X_train_sc, X_explain, model_name)
        shap_stability = compute_shap_stability(model, X_explain, shap_vals, model_name, X_train_sc)
        # SHAP consistency: run TreeSHAP twice (it's deterministic); for KernelSHAP measure across 2 runs
        if model_name == "Random Forest":
            shap_consistency = 1.0   # TreeSHAP is deterministic
            shap_consistency_std = 0.0
        else:
            # Run KernelSHAP twice, compare
            background = shap.kmeans(X_train_sc.values, 10)
            exp1 = shap.KernelExplainer(model.predict_proba, background)
            v1   = np.abs(exp1.shap_values(X_explain.values[:10], nsamples=100))
            v1   = v1[1] if isinstance(v1, list) else v1
            v2   = np.abs(exp1.shap_values(X_explain.values[:10], nsamples=100))
            v2   = v2[1] if isinstance(v2, list) else v2
            shap_consistency_std = float(np.abs(v1 - v2).mean())
            shap_consistency     = float(1 - min(shap_consistency_std, 1.0))
        print(f"consistency={shap_consistency:.4f}, stability={shap_stability:.4f}")

        # ── LIME ──────────────────────────────────
        print("     Computing LIME...", end=" ", flush=True)
        lime_consistency, lime_mean_std, lime_vals = compute_lime_consistency(
            model, X_train_sc, X_explain, feature_names
        )
        lime_stability = compute_lime_stability(model, X_train_sc, X_explain, feature_names)
        print(f"consistency={lime_consistency:.4f}, stability={lime_stability:.4f}")

        # ── Rank correlation vs LR baseline ───────
        rho_shap, p_shap, rho_lime, p_lime = compute_rank_correlation(shap_vals, lime_vals, lr_importances)

        elapsed = time.time() - t0
        print(f"     Done in {elapsed:.1f}s")

        all_results.append({
            "Dataset":            dataset_name,
            "Model":              model_name,
            "Accuracy":           round(acc,  4),
            "F1 Score":           round(f1,   4),
            "SHAP Consistency":   round(shap_consistency, 4),
            "SHAP Stability":     round(shap_stability,   4),
            "LIME Consistency":   round(lime_consistency, 4),
            "LIME Stability":     round(lime_stability,   4),
            "Rho SHAP vs LR":     round(rho_shap, 4),
            "Rho LIME vs LR":     round(rho_lime, 4),
            "p SHAP":             round(p_shap, 4),
            "p LIME":             round(p_lime, 4),
        })

# ─────────────────────────────────────────────
# 5. SAVE RESULTS TABLE
# ─────────────────────────────────────────────

results_df = pd.DataFrame(all_results)
csv_path   = os.path.join(OUTPUT_DIR, "xai_metrics_table.csv")
results_df.to_csv(csv_path, index=False)
print(f"\n\nResults saved → {csv_path}")
print(results_df.to_string(index=False))

# ─────────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {"SHAP": "#2196F3", "LIME": "#FF5722"}

def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

datasets_list = results_df["Dataset"].unique()
models_list   = results_df["Model"].unique()

# ── Figure 1: Accuracy & F1 per dataset ──────
fig, axes = plt.subplots(1, len(datasets_list), figsize=(6*len(datasets_list), 5), sharey=False)
if len(datasets_list) == 1:
    axes = [axes]
for ax, ds in zip(axes, datasets_list):
    sub = results_df[results_df["Dataset"] == ds]
    x   = np.arange(len(sub))
    w   = 0.35
    ax.bar(x - w/2, sub["Accuracy"], w, label="Accuracy", color="#4CAF50")
    ax.bar(x + w/2, sub["F1 Score"], w, label="F1 Score",  color="#9C27B0")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["Model"], rotation=15, ha="right")
    ax.set_ylim(0.5, 1.05)
    ax.set_title(ds)
    ax.set_ylabel("Score")
    ax.legend()
plt.suptitle("Model Performance Across Datasets", fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("fig1_model_performance.png")

# ── Figure 2: Consistency comparison ─────────
fig, axes = plt.subplots(1, len(datasets_list), figsize=(6*len(datasets_list), 5), sharey=True)
if len(datasets_list) == 1:
    axes = [axes]
for ax, ds in zip(axes, datasets_list):
    sub = results_df[results_df["Dataset"] == ds]
    x   = np.arange(len(sub))
    w   = 0.35
    ax.bar(x - w/2, sub["SHAP Consistency"], w, label="SHAP", color=COLORS["SHAP"])
    ax.bar(x + w/2, sub["LIME Consistency"], w, label="LIME", color=COLORS["LIME"])
    ax.set_xticks(x)
    ax.set_xticklabels(sub["Model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, ls="--", color="gray", lw=0.8)
    ax.set_title(ds)
    ax.set_ylabel("Consistency Score (↑ better)")
    ax.legend()
plt.suptitle("SHAP vs LIME: Consistency Across Datasets", fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("fig2_consistency_comparison.png")

# ── Figure 3: Stability comparison ───────────
fig, axes = plt.subplots(1, len(datasets_list), figsize=(6*len(datasets_list), 5), sharey=True)
if len(datasets_list) == 1:
    axes = [axes]
for ax, ds in zip(axes, datasets_list):
    sub = results_df[results_df["Dataset"] == ds]
    x   = np.arange(len(sub))
    w   = 0.35
    ax.bar(x - w/2, sub["SHAP Stability"], w, label="SHAP", color=COLORS["SHAP"])
    ax.bar(x + w/2, sub["LIME Stability"], w, label="LIME", color=COLORS["LIME"])
    ax.set_xticks(x)
    ax.set_xticklabels(sub["Model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, ls="--", color="gray", lw=0.8)
    ax.set_title(ds)
    ax.set_ylabel("Stability Score (↑ better)")
    ax.legend()
plt.suptitle("SHAP vs LIME: Stability Under Input Perturbation", fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("fig3_stability_comparison.png")

# ── Figure 4: Rank correlation heatmap ───────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, method, col in zip(axes, ["SHAP", "LIME"], ["Rho SHAP vs LR", "Rho LIME vs LR"]):
    pivot = results_df.pivot(index="Model", columns="Dataset", values=col)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                cbar_kws={"label": "Spearman ρ"})
    ax.set_title(f"{method} – Rank Correlation vs LR (ground truth)")
    ax.set_xlabel("")
plt.suptitle("Feature Importance Rank Correlation with Logistic Regression Baseline",
             fontweight="bold", y=1.02)
plt.tight_layout()
save_fig("fig4_rank_correlation_heatmap.png")

# ── Figure 5: Radar / spider chart ───────────
# Aggregate mean metrics across datasets per model
metrics_cols = ["SHAP Consistency", "SHAP Stability", "LIME Consistency", "LIME Stability"]
agg = results_df.groupby("Model")[metrics_cols].mean()

labels   = ["SHAP\nConsistency", "SHAP\nStability", "LIME\nConsistency", "LIME\nStability"]
n_labels = len(labels)
angles   = np.linspace(0, 2*np.pi, n_labels, endpoint=False).tolist()
angles  += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
model_colors = ["#E91E63", "#00BCD4", "#FF9800"]
for (mname, row), col in zip(agg.iterrows(), model_colors):
    vals  = row.tolist() + [row.tolist()[0]]
    ax.plot(angles, vals,  "o-", lw=2,   color=col, label=mname)
    ax.fill(angles, vals, alpha=0.1, color=col)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0, 1)
ax.set_title("XAI Quality Radar (avg across datasets)", fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
save_fig("fig5_radar_chart.png")

# ── Figure 6: SHAP summary plot (Breast Cancer / Random Forest) ──
print("\nGenerating SHAP summary plot for Breast Cancer / Random Forest...")
X_bc, y_bc, _ = load_breast_cancer_data()
fn_bc          = list(X_bc.columns)
Xtr, Xte, ytr, yte = train_test_split(X_bc, y_bc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_bc)
sc2 = StandardScaler()
Xtr_sc = pd.DataFrame(sc2.fit_transform(Xtr), columns=fn_bc)
Xte_sc = pd.DataFrame(sc2.transform(Xte),     columns=fn_bc)
rf2    = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf2.fit(Xtr_sc, ytr)
explainer_bc = shap.TreeExplainer(rf2)
sv_bc        = explainer_bc.shap_values(Xte_sc.values)
sv_bc_class1 = sv_bc[1] if isinstance(sv_bc, list) else sv_bc
plt.figure()
shap.summary_plot(sv_bc_class1, Xte_sc, feature_names=fn_bc, show=False, max_display=15)
plt.title("SHAP Summary – Breast Cancer / Random Forest", fontweight="bold")
save_fig("fig6_shap_summary_breast_cancer.png")

# ─────────────────────────────────────────────
# 7. PRINT PAPER-READY SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("PAPER-READY SUMMARY TABLE")
print("="*60)
summary_cols = ["Dataset", "Model", "Accuracy", "F1 Score",
                "SHAP Consistency", "LIME Consistency",
                "SHAP Stability",   "LIME Stability",
                "Rho SHAP vs LR",   "Rho LIME vs LR"]
print(results_df[summary_cols].to_string(index=False))

print("\n\nALL FIGURES & CSV SAVED TO:", os.path.abspath(OUTPUT_DIR))
print("Files:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {f}")