"""
=============================================================
  QUANT STRATEGY: Machine Learning Alpha
  Layer 1: Feature Engineering
=============================================================
  Requires: prices_daily.csv (from momentum project Layer 1)
  We reuse the same price data — no need to re-download.
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def compute_features(prices_monthly, prices_daily):
    """
    Build the feature matrix: one row per (stock, month).
    Every feature is computed using ONLY past data — no lookahead.

    Features fall into 4 categories:
    1. Momentum      — trend following signals
    2. Reversal      — mean reversion signals
    3. Volatility    — risk signals
    4. Technical     — price structure signals

    The TARGET variable is next month's return — what we predict.

    CRITICAL RULE: Every feature at time T must use only
    data available at time T or earlier. If you accidentally
    use T+1 data in a feature, your model is cheating.
    """

    features_list = []

    print("[Layer 1] Computing features for each stock/month...")

    # Monthly returns (used in many features)
    monthly_ret = prices_monthly.pct_change()

    # Daily returns for vol calculations
    daily_ret = prices_daily.pct_change()

    for i, date in enumerate(prices_monthly.index[24:], start=24):
        if i % 24 == 0:
            print(f"  Processing {date.strftime('%Y-%m')}...")

        for ticker in prices_monthly.columns:
            try:
                # Get price history up to this month
                p_monthly = prices_monthly[ticker].iloc[:i+1].dropna()
                if len(p_monthly) < 13:
                    continue

                current_price = p_monthly.iloc[-1]
                if pd.isna(current_price) or current_price <= 0:
                    continue

                row = {"date": date, "ticker": ticker}

                # ── MOMENTUM FEATURES ────────────────────
                # 1-month return (will show reversal, not momentum)
                row["ret_1m"]  = p_monthly.iloc[-1] / p_monthly.iloc[-2]  - 1

                # 12-1 momentum (standard)
                row["ret_12_1"] = p_monthly.iloc[-2] / p_monthly.iloc[-13] - 1

                # 6-1 momentum
                if len(p_monthly) >= 7:
                    row["ret_6_1"] = p_monthly.iloc[-2] / p_monthly.iloc[-7] - 1

                # 3-1 momentum
                if len(p_monthly) >= 4:
                    row["ret_3_1"] = p_monthly.iloc[-2] / p_monthly.iloc[-4] - 1

                # 24-month momentum (longer term)
                if len(p_monthly) >= 25:
                    row["ret_24m"] = p_monthly.iloc[-1] / p_monthly.iloc[-25] - 1

                # ── VOLATILITY FEATURES ──────────────────
                # Get daily prices for this ticker
                if ticker in daily_ret.columns:
                    d_ret = daily_ret[ticker].loc[:date].dropna()

                    # 1-month realized vol (last 21 trading days)
                    if len(d_ret) >= 21:
                        row["vol_1m"] = d_ret.iloc[-21:].std() * np.sqrt(252)

                    # 3-month realized vol
                    if len(d_ret) >= 63:
                        row["vol_3m"] = d_ret.iloc[-63:].std() * np.sqrt(252)

                    # Vol ratio: recent vol / longer vol (vol trend)
                    if len(d_ret) >= 63:
                        row["vol_ratio"] = (d_ret.iloc[-21:].std() /
                                           d_ret.iloc[-63:].std()
                                           if d_ret.iloc[-63:].std() > 0 else 1)

                # ── TECHNICAL FEATURES ───────────────────
                # 52-week high ratio: how close to yearly high?
                # High ratio = strong momentum, stock near highs
                if len(p_monthly) >= 12:
                    high_52w = p_monthly.iloc[-12:].max()
                    row["price_to_52w_high"] = current_price / high_52w

                # Distance from 12-month SMA (normalized)
                if len(p_monthly) >= 12:
                    sma_12 = p_monthly.iloc[-12:].mean()
                    row["dist_from_sma12"] = (current_price - sma_12) / sma_12

                # Distance from 6-month SMA
                if len(p_monthly) >= 6:
                    sma_6 = p_monthly.iloc[-6:].mean()
                    row["dist_from_sma6"] = (current_price - sma_6) / sma_6

                # ── TARGET VARIABLE ──────────────────────
                # Next month's return — what we're predicting
                # This uses future data ONLY for the target,
                # never for features. This is correct.
                if i + 1 < len(prices_monthly):
                    next_price = prices_monthly[ticker].iloc[i+1]
                    if not pd.isna(next_price) and current_price > 0:
                        row["target"] = next_price / current_price - 1
                    else:
                        row["target"] = np.nan
                else:
                    row["target"] = np.nan

                features_list.append(row)

            except Exception:
                continue

    df = pd.DataFrame(features_list)
    df = df.set_index(["date", "ticker"])
    print(f"\n[Layer 1] Raw feature matrix: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# CLEAN FEATURE MATRIX
# ─────────────────────────────────────────────────────────────

def clean_features(df):
    """
    Clean the feature matrix before feeding to ML models.

    Steps:
    1. Drop rows with missing target (can't train on these)
    2. Drop rows with >30% missing features
    3. Fill remaining NaNs with cross-sectional median
       (median per month, not overall — prevents lookahead)
    4. Winsorize extreme values (cap at 1st/99th percentile)
       Prevents one crazy outlier from dominating the model
    """
    print("\n[Layer 1] Cleaning feature matrix...")

    # Drop missing targets
    df = df.dropna(subset=["target"])
    print(f"  After dropping missing targets: {df.shape}")

    # Feature columns only
    feature_cols = [c for c in df.columns if c != "target"]

    # Drop rows with too many missing features
    thresh = int(len(feature_cols) * 0.70)
    df = df.dropna(thresh=thresh)
    print(f"  After missing feature filter:   {df.shape}")

    # Fill remaining NaN with cross-sectional median per month
    def fill_cs_median(group):
        return group.fillna(group.median())

    df[feature_cols] = df.groupby(level="date")[feature_cols].transform(
        lambda x: x.fillna(x.median())
    )

    # Winsorize at 1st/99th percentile
    for col in feature_cols:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    df = df.dropna()
    print(f"  Final clean shape:              {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# FEATURE DIAGNOSTICS
# ─────────────────────────────────────────────────────────────

def feature_diagnostics(df):
    """
    Check if features have any predictive power BEFORE training.
    Compute Information Coefficient (IC) for each feature.

    IC = rank correlation between feature and next month's return.
    IC > 0.02 = weak but tradeable signal
    IC > 0.05 = strong signal
    IC < 0    = feature hurts more than helps

    This is standard practice at quant funds before any ML.
    """
    feature_cols = [c for c in df.columns if c != "target"]

    print("\n" + "="*55)
    print("  FEATURE INFORMATION COEFFICIENTS (IC)")
    print("  IC = rank correlation with next month return")
    print("="*55)

    ics = {}
    for col in feature_cols:
        monthly_ics = []
        for date, group in df.groupby(level="date"):
            if len(group) < 10:
                continue
            ic = group[col].rank().corr(group["target"].rank(),
                                         method="spearman")
            monthly_ics.append(ic)
        mean_ic = np.mean(monthly_ics)
        ic_std  = np.std(monthly_ics)
        icir    = mean_ic / ic_std if ic_std > 0 else 0  # IC Information Ratio
        ics[col] = {"IC": round(mean_ic, 4), "IC_std": round(ic_std, 4),
                    "ICIR": round(icir, 4)}
        signal = "STRONG" if abs(mean_ic) > 0.04 else ("WEAK" if abs(mean_ic) > 0.01 else "NOISE")
        print(f"  {col:<22} IC: {mean_ic:+.4f}  ICIR: {icir:+.3f}  [{signal}]")

    print("="*55)
    return pd.DataFrame(ics).T


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_features(df, ic_df):
    """
    3-panel chart:
    1. Feature IC bar chart (which features are predictive)
    2. Feature correlation heatmap (check for redundancy)
    3. Target distribution (what we're trying to predict)
    """
    BG, PANEL    = "#0f0f0f", "#1a1a1a"
    WHITE        = "#e8e8e8"
    GREEN, RED   = "#00ff88", "#ff4466"
    YELLOW, BLUE = "#f5c518", "#4488ff"

    fig = plt.figure(figsize=(16, 14), facecolor=BG)
    gs  = gridspec.GridSpec(3, 1, hspace=0.55)

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold", pad=7)
        ax.tick_params(colors=WHITE, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.grid(True, color="#222222", linewidth=0.5, alpha=0.8)

    # ── Panel 1: IC Bar Chart ─────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ic_sorted = ic_df["IC"].sort_values()
    colors    = [GREEN if v > 0 else RED for v in ic_sorted.values]
    bars = ax1.barh(ic_sorted.index, ic_sorted.values,
                    color=colors, alpha=0.85, edgecolor="none")
    ax1.axvline(0,     color=WHITE,  lw=1)
    ax1.axvline(0.02,  color=YELLOW, lw=1, linestyle="--", alpha=0.6, label="IC=0.02 threshold")
    ax1.axvline(-0.02, color=YELLOW, lw=1, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)
    style_ax(ax1, "Feature Information Coefficients (IC)  |  Green = Positive Predictive Power")

    # ── Panel 2: Feature Correlation Heatmap ─────────────
    ax2 = fig.add_subplot(gs[1])
    feature_cols = [c for c in df.columns if c != "target"]
    corr = df[feature_cols].corr()
    im   = ax2.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(corr.columns)))
    ax2.set_yticks(range(len(corr.columns)))
    ax2.set_xticklabels(corr.columns, rotation=45, ha="right",
                        color=WHITE, fontsize=7)
    ax2.set_yticklabels(corr.columns, color=WHITE, fontsize=7)
    plt.colorbar(im, ax=ax2)
    style_ax(ax2, "Feature Correlation Matrix  |  Highly correlated features = redundant")

    # ── Panel 3: Target Distribution ─────────────────────
    ax3 = fig.add_subplot(gs[2])
    target = df["target"].clip(-0.3, 0.3) * 100
    ax3.hist(target[target > 0], bins=60, color=GREEN,
             alpha=0.7, label=f"Positive ({(target>0).sum()})", edgecolor="none")
    ax3.hist(target[target <= 0], bins=60, color=RED,
             alpha=0.7, label=f"Negative ({(target<=0).sum()})", edgecolor="none")
    ax3.axvline(0, color=WHITE, lw=1)
    ax3.axvline(target.mean(), color=YELLOW, lw=1.5, linestyle="--",
                label=f"Mean: {target.mean():.2f}%")
    ax3.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)
    style_ax(ax3, "Target Variable Distribution (Next Month Return %)  |  What the model predicts")

    fig.suptitle("ML Alpha — Feature Engineering Analysis (Layer 1)",
                 color=WHITE, fontsize=13, fontweight="bold", y=0.999)

    plt.savefig("ml_features.png", dpi=150, bbox_inches="tight", facecolor=BG)
    print("[Layer 1] Chart saved -> ml_features.png")
    plt.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Load price data from momentum project
    print("[Layer 1] Loading price data...")
    prices_daily   = pd.read_csv("prices_daily.csv",   index_col=0, parse_dates=True)
    prices_monthly = pd.read_csv("prices_monthly.csv", index_col=0, parse_dates=True)
    print(f"  Daily  : {prices_daily.shape}")
    print(f"  Monthly: {prices_monthly.shape}")

    # Build feature matrix (~2-3 minutes)
    print("\n[Layer 1] Building feature matrix (2-3 minutes)...")
    features = compute_features(prices_monthly, prices_daily)

    # Clean
    features = clean_features(features)

    # Diagnostics — IC analysis
    ic_df = feature_diagnostics(features)

    # Visualize
    plot_features(features, ic_df)

    # Save
    features.to_csv("ml_features.csv")
    print("\n[Done] Saved -> ml_features.csv")
    print(f"\nFinal dataset: {features.shape[0]:,} observations, "
          f"{features.shape[1]-1} features + 1 target")
