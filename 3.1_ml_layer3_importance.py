"""
=============================================================
  QUANT STRATEGY: Machine Learning Alpha
  Layer 3: Feature Importance + Model Interpretability
=============================================================
  Requires: ml_features.csv, predictions_*.csv
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_XGB = False


# ─────────────────────────────────────────────────────────────
# TRAIN FINAL MODELS ON FULL DATASET
# ─────────────────────────────────────────────────────────────

def train_final_models(df):
    """
    Train each model on the FULL dataset to extract feature importance.

    NOTE: This is for interpretability only — not for trading signals.
    We already have out-of-sample results from Layer 2.
    Training on full data here just gives us stable importance estimates.
    """
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values
    y = df["target"].values

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Ridge — importance = absolute coefficient value
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    ridge_importance = pd.Series(
        np.abs(ridge.coef_), index=feature_cols
    ).sort_values(ascending=False)

    # Random Forest — importance = mean decrease in impurity
    rf = RandomForestRegressor(n_estimators=200, max_depth=5,
                                min_samples_leaf=20, random_state=42,
                                n_jobs=-1)
    rf.fit(X_scaled, y)
    rf_importance = pd.Series(
        rf.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    # XGBoost — importance = gain (how much each feature improves splits)
    if HAS_XGB:
        xgb = XGBRegressor(n_estimators=200, max_depth=4,
                           learning_rate=0.05, subsample=0.8,
                           random_state=42, verbosity=0)
        xgb.fit(X_scaled, y)
        xgb_importance = pd.Series(
            xgb.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                        learning_rate=0.05, random_state=42)
        gb.fit(X_scaled, y)
        xgb_importance = pd.Series(
            gb.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)

    return ridge_importance, rf_importance, xgb_importance, feature_cols


# ─────────────────────────────────────────────────────────────
# PERMUTATION IMPORTANCE
# ─────────────────────────────────────────────────────────────

def permutation_importance(df, n_repeats=10):
    """
    Permutation Importance — the most honest importance measure.

    Method:
    1. Train model, record baseline IC
    2. Shuffle one feature column (break its relationship with target)
    3. Measure how much IC drops
    4. Drop = how much that feature contributes

    WHY THIS IS BETTER than built-in importance:
    Built-in RF importance can overrate high-cardinality features.
    Permutation importance directly measures predictive contribution.
    This is what serious quant researchers use.
    """
    print("\n[Layer 3] Computing permutation importance (takes ~2 min)...")
    feature_cols = [c for c in df.columns if c != "target"]

    # Use last 3 years for speed
    dates    = sorted(df.index.get_level_values("date").unique())
    cutoff   = dates[int(len(dates) * 0.6)]
    train_df = df.loc[df.index.get_level_values("date") < cutoff]
    test_df  = df.loc[df.index.get_level_values("date") >= cutoff]

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["target"].values

    scaler  = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Train RF for permutation test
    rf = RandomForestRegressor(n_estimators=100, max_depth=5,
                                min_samples_leaf=20, random_state=42,
                                n_jobs=-1)
    rf.fit(X_train, y_train)

    # Baseline IC
    baseline_preds = rf.predict(X_test)
    baseline_ic, _ = spearmanr(baseline_preds, y_test)

    # Permute each feature
    perm_drops = {}
    for i, col in enumerate(feature_cols):
        drops = []
        for _ in range(n_repeats):
            X_permuted      = X_test.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            perm_preds      = rf.predict(X_permuted)
            perm_ic, _      = spearmanr(perm_preds, y_test)
            drops.append(baseline_ic - perm_ic)
        perm_drops[col] = np.mean(drops)
        print(f"  {col:<22} IC drop: {np.mean(drops):+.4f}")

    return pd.Series(perm_drops).sort_values(ascending=False), baseline_ic


# ─────────────────────────────────────────────────────────────
# IC STABILITY OVER TIME
# ─────────────────────────────────────────────────────────────

def ic_stability(df):
    """
    Compute IC for each feature by year.
    A good feature should be consistently predictive,
    not just work in one regime.

    If a feature only works 2010-2015 but not after,
    it's likely a historical artifact, not a real signal.
    """
    feature_cols = [c for c in df.columns if c != "target"]
    dates        = df.index.get_level_values("date")
    years        = sorted(dates.year.unique())

    results = {}
    for col in feature_cols:
        yearly_ics = {}
        for yr in years:
            yr_data = df.loc[dates.year == yr]
            if len(yr_data) < 20:
                continue
            ic, _ = spearmanr(yr_data[col], yr_data["target"])
            yearly_ics[yr] = ic if not np.isnan(ic) else 0
        results[col] = yearly_ics

    return pd.DataFrame(results).T


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_importance(ridge_imp, rf_imp, xgb_imp, perm_imp,
                    ic_stability_df, baseline_ic):
    BG, PANEL    = "#0f0f0f", "#1a1a1a"
    WHITE        = "#e8e8e8"
    GREEN, RED   = "#00ff88", "#ff4466"
    YELLOW, BLUE = "#f5c518", "#4488ff"
    PURPLE       = "#cc88ff"

    fig = plt.figure(figsize=(16, 20), facecolor=BG)
    gs  = gridspec.GridSpec(5, 1, hspace=0.55)

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold", pad=7)
        ax.tick_params(colors=WHITE, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.grid(True, color="#222222", linewidth=0.5, alpha=0.8)

    # ── Panel 1: Ridge Coefficients ───────────────────────
    ax1 = fig.add_subplot(gs[0])
    colors = [GREEN if v > 0 else RED for v in ridge_imp.values]
    ax1.barh(ridge_imp.index, ridge_imp.values,
             color=colors, alpha=0.85, edgecolor="none")
    ax1.axvline(0, color=WHITE, lw=1)
    style_ax(ax1, "Ridge Regression — Feature Coefficients  "
                  "|  Larger = More Important to Linear Model")

    # ── Panel 2: Random Forest Importance ─────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.barh(rf_imp.index, rf_imp.values,
             color=BLUE, alpha=0.85, edgecolor="none")
    ax2.axvline(0, color=WHITE, lw=1)
    style_ax(ax2, "Random Forest — Feature Importance (Mean Decrease Impurity)")

    # ── Panel 3: XGBoost Importance ───────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.barh(xgb_imp.index, xgb_imp.values,
             color=GREEN, alpha=0.85, edgecolor="none")
    style_ax(ax3, "XGBoost — Feature Importance (Gain)  "
                  "|  The Most Reliable Tree-Based Measure")

    # ── Panel 4: Permutation Importance ───────────────────
    ax4 = fig.add_subplot(gs[3])
    colors = [GREEN if v > 0 else RED for v in perm_imp.values]
    ax4.barh(perm_imp.index, perm_imp.values,
             color=colors, alpha=0.85, edgecolor="none")
    ax4.axvline(0, color=WHITE, lw=1)
    ax4.set_xlabel(f"IC Drop when Permuted  (Baseline IC: {baseline_ic:.4f})",
                   color=WHITE, fontsize=8)
    style_ax(ax4, "Permutation Importance — Most Honest Measure  "
                  "|  How Much IC Drops When Feature is Shuffled")

    # ── Panel 5: IC Stability Heatmap ─────────────────────
    ax5 = fig.add_subplot(gs[4])
    im = ax5.imshow(ic_stability_df.values, cmap="RdYlGn",
                    vmin=-0.05, vmax=0.05, aspect="auto")
    ax5.set_xticks(range(len(ic_stability_df.columns)))
    ax5.set_xticklabels(ic_stability_df.columns.tolist(),
                        color=WHITE, fontsize=8, rotation=45)
    ax5.set_yticks(range(len(ic_stability_df.index)))
    ax5.set_yticklabels(ic_stability_df.index.tolist(),
                        color=WHITE, fontsize=8)
    plt.colorbar(im, ax=ax5, label="IC")
    style_ax(ax5, "Feature IC by Year  |  Consistent Green = Stable Signal Across Regimes")

    fig.suptitle(
        "ML Alpha — Feature Importance & Interpretability (Layer 3)",
        color=WHITE, fontsize=13, fontweight="bold", y=0.999)

    plt.savefig("ml_feature_importance.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    print("[Layer 3] Chart saved -> ml_feature_importance.png")
    plt.close()


# ─────────────────────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────────────────────

def print_importance_summary(ridge_imp, rf_imp, xgb_imp, perm_imp):
    print(f"\n{'='*62}")
    print(f"  FEATURE IMPORTANCE SUMMARY")
    print(f"{'='*62}")
    print(f"  {'Feature':<22} {'Ridge':>8} {'RF':>8} "
          f"{'XGB':>8} {'Perm':>8}")
    print(f"  {'-'*56}")

    # Normalize each to sum to 1 for comparison
    r = ridge_imp / ridge_imp.sum()
    f = rf_imp    / rf_imp.sum()
    x = xgb_imp   / xgb_imp.sum()
    p = perm_imp  / perm_imp.abs().sum()

    all_features = ridge_imp.index.tolist()
    for feat in all_features:
        print(f"  {feat:<22} {r.get(feat,0):>8.3f} "
              f"{f.get(feat,0):>8.3f} "
              f"{x.get(feat,0):>8.3f} "
              f"{p.get(feat,0):>+8.3f}")
    print(f"{'='*62}")

    print(f"\n  Top features by permutation importance:")
    top = perm_imp.head(5)
    for feat, val in top.items():
        print(f"    {feat:<22} IC drop: {val:+.4f}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("[Layer 3] Loading features...")
    df = pd.read_csv("ml_features.csv",
                     index_col=["date","ticker"], parse_dates=["date"])
    print(f"  Shape: {df.shape}")

    # 1. Train final models + get importance
    print("\n[Layer 3] Training final models for importance extraction...")
    ridge_imp, rf_imp, xgb_imp, feature_cols = train_final_models(df)

    # 2. Permutation importance (most honest)
    perm_imp, baseline_ic = permutation_importance(df, n_repeats=5)

    # 3. IC stability by year
    print("\n[Layer 3] Computing IC stability by year...")
    ic_stab = ic_stability(df)

    # 4. Print summary
    print_importance_summary(ridge_imp, rf_imp, xgb_imp, perm_imp)

    # 5. Plot
    plot_importance(ridge_imp, rf_imp, xgb_imp, perm_imp,
                    ic_stab, baseline_ic)

    print("\n[Done] Feature importance analysis complete.")
    print("  -> ml_feature_importance.png")
