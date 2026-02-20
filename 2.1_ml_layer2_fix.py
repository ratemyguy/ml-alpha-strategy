
"""
=============================================================
  QUANT STRATEGY: Machine Learning Alpha
  Layer 2 FIX: Corrected Portfolio Simulation
=============================================================
  Requires: ml_features.csv, prices_monthly.csv
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# WALK-FORWARD ENGINE
# ─────────────────────────────────────────────────────────────

def walk_forward_predict(df, model, train_years=4, name="Model"):
    """
    Walk-forward validation. Returns a DataFrame with
    date, ticker, predicted return, actual return.
    """
    feature_cols = [c for c in df.columns if c != "target"]
    dates        = sorted(df.index.get_level_values("date").unique())
    records      = []
    monthly_ics  = []

    print(f"\n  [{name}] Walk-forward validation...")

    for i, test_date in enumerate(dates):
        train_dates = [d for d in dates if d < test_date]
        if len(train_dates) < train_years * 12:
            continue

        train_data = df.loc[df.index.get_level_values("date").isin(train_dates)]
        test_data  = df.loc[df.index.get_level_values("date") == test_date]

        if len(test_data) < 5:
            continue

        X_train = train_data[feature_cols].values
        y_train = train_data["target"].values
        X_test  = test_data[feature_cols].values
        y_test  = test_data["target"].values

        scaler  = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        except Exception:
            continue

        # Store date + ticker + prediction + actual
        tickers = test_data.index.get_level_values("ticker")
        for ticker, pred, actual in zip(tickers, preds, y_test):
            records.append({
                "date":      test_date,
                "ticker":    ticker,
                "predicted": pred,
                "actual":    actual,
            })

        ic, _ = spearmanr(preds, y_test)
        if not np.isnan(ic):
            monthly_ics.append(ic)

        if i % 12 == 0 and monthly_ics:
            print(f"    {test_date.strftime('%Y-%m')}  "
                  f"IC: {np.mean(monthly_ics):.4f}  "
                  f"months: {len(monthly_ics)}")

    results_df = pd.DataFrame(records)
    mean_ic = np.mean(monthly_ics) if monthly_ics else 0
    icir    = mean_ic / np.std(monthly_ics) if np.std(monthly_ics) > 0 else 0
    print(f"    Final Mean IC: {mean_ic:.4f}  ICIR: {icir:.3f}")
    return results_df, monthly_ics


# ─────────────────────────────────────────────────────────────
# PORTFOLIO SIMULATION
# ─────────────────────────────────────────────────────────────

def simulate_portfolio(pred_df, prices, top_pct=0.20, cost=0.001):
    """
    Each month: buy top 20% by predicted return, equal weight.
    """
    monthly_ret  = prices.pct_change()
    port_returns = []
    port_dates   = []
    prev_held    = set()

    for date, group in pred_df.groupby("date"):
        # Get next month's actual returns
        future_dates = [d for d in monthly_ret.index if d > date]
        if len(future_dates) == 0:
            continue
        next_date = future_dates[0]

        if next_date not in monthly_ret.index:
            continue

        # Rank by predicted return, take top 20%
        n_top    = max(1, int(len(group) * top_pct))
        top_rows = group.nlargest(n_top, "predicted")
        curr_held = set(top_rows["ticker"].tolist())

        # Get actual returns for held stocks
        next_rets = monthly_ret.loc[next_date]
        valid_held = curr_held & set(next_rets.dropna().index)

        if len(valid_held) == 0:
            port_returns.append(0)
            port_dates.append(next_date)
            prev_held = curr_held
            continue

        port_ret = next_rets[list(valid_held)].mean()

        # Transaction costs
        if len(prev_held) > 0:
            entered  = curr_held - prev_held
            exited   = prev_held - curr_held
            turnover = (len(entered) + len(exited)) / (2 * len(curr_held))
        else:
            turnover = 1.0

        net_ret = port_ret - turnover * cost * 2
        port_returns.append(net_ret)
        port_dates.append(next_date)
        prev_held = curr_held

    return pd.Series(port_returns, index=port_dates).sort_index()


# ─────────────────────────────────────────────────────────────
# METRICS + VISUALIZATION
# ─────────────────────────────────────────────────────────────

def compute_metrics(returns, label=""):
    r = returns.dropna()
    if len(r) < 6:
        return {}
    equity   = (1 + r).cumprod()
    years    = len(r) / 12
    ann_ret  = (equity.iloc[-1] ** (1/years) - 1) * 100
    ann_vol  = r.std() * np.sqrt(12) * 100
    sharpe   = (r.mean() * 12) / (r.std() * np.sqrt(12)) if r.std() > 0 else 0
    roll_max = equity.cummax()
    max_dd   = ((equity - roll_max) / roll_max * 100).min()
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else 0
    downside = r[r < 0]
    sortino  = (r.mean() * 12) / (downside.std() * np.sqrt(12)) if len(downside) > 1 else 0

    print(f"\n{'='*52}")
    print(f"  PERFORMANCE — {label}")
    print(f"{'='*52}")
    print(f"  Ann. Return    : {ann_ret:.2f}%")
    print(f"  Ann. Volatility: {ann_vol:.2f}%")
    print(f"  Sharpe Ratio   : {sharpe:.3f}")
    print(f"  Sortino Ratio  : {sortino:.3f}")
    print(f"  Max Drawdown   : {max_dd:.2f}%")
    print(f"  Calmar Ratio   : {calmar:.3f}")
    print(f"  Final Equity   : ${100000*equity.iloc[-1]:,.0f}")
    print(f"{'='*52}")

    return {"label": label, "ann_ret": ann_ret, "ann_vol": ann_vol,
            "sharpe": sharpe, "sortino": sortino, "max_dd": max_dd,
            "calmar": calmar, "equity": equity, "returns": r}


def plot_results(all_metrics, all_ics, benchmark):
    BG, PANEL    = "#0f0f0f", "#1a1a1a"
    WHITE        = "#e8e8e8"
    GREEN, RED   = "#00ff88", "#ff4466"
    YELLOW, BLUE = "#f5c518", "#4488ff"
    PURPLE       = "#cc88ff"
    COLORS       = [YELLOW, BLUE, GREEN]

    fig = plt.figure(figsize=(16, 16), facecolor=BG)
    gs  = gridspec.GridSpec(4, 1, hspace=0.50)

    def style_ax(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=WHITE, fontsize=10, fontweight="bold", pad=7)
        ax.tick_params(colors=WHITE, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.grid(True, color="#222222", linewidth=0.5, alpha=0.8)

    # Panel 1: Equity Curves
    ax1 = fig.add_subplot(gs[0])
    for m, c in zip(all_metrics, COLORS):
        eq = m["equity"] * 100000
        ax1.plot(eq.index, eq, color=c, lw=1.5,
                 label=f"{m['label']}  Sharpe:{m['sharpe']:.2f}  Ann:{m['ann_ret']:.1f}%")
    if benchmark is not None:
        b_eq = (1 + benchmark.dropna()).cumprod() * 100000
        ax1.plot(b_eq.index, b_eq, color=WHITE, lw=1,
                 linestyle="--", alpha=0.5, label="Equal-Weight Benchmark")
    ax1.axhline(100000, color=WHITE, lw=0.4, linestyle=":", alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}k"))
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)
    style_ax(ax1, "Equity Curves — ML Models vs Benchmark  (Walk-Forward Out-of-Sample)")

    # Panel 2: Rolling IC
    ax2 = fig.add_subplot(gs[1])
    for (name, ics), c in zip(all_ics.items(), COLORS):
        s = pd.Series(ics)
        ax2.plot(range(len(s)), s, color=c, lw=0.6, alpha=0.4)
        ax2.plot(range(len(s)), s.rolling(12).mean(), color=c, lw=2, label=name)
    ax2.axhline(0,    color=WHITE,  lw=0.8, linestyle="--")
    ax2.axhline(0.05, color=YELLOW, lw=0.8, linestyle=":", alpha=0.5)
    ax2.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)
    style_ax(ax2, "Monthly IC — Raw (thin) + 12m Rolling (thick)  |  Positive = Predictive")

    # Panel 3: Sharpe bar
    ax3 = fig.add_subplot(gs[2])
    names   = [m["label"]   for m in all_metrics]
    sharpes = [m["sharpe"]  for m in all_metrics]
    bars = ax3.bar(names, sharpes, color=COLORS[:len(names)],
                   alpha=0.85, edgecolor="none")
    ax3.axhline(0, color=WHITE, lw=0.8)
    ax3.axhline(1, color=YELLOW, lw=1, linestyle="--", alpha=0.6, label="Sharpe=1")
    for bar, val in zip(bars, sharpes):
        ax3.text(bar.get_x() + bar.get_width()/2,
                 val + (0.02 if val >= 0 else -0.08),
                 f"{val:.2f}", ha="center", color=WHITE, fontsize=10)
    ax3.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)
    style_ax(ax3, "Out-of-Sample Sharpe Ratio by Model")

    # Panel 4: Drawdown
    ax4 = fig.add_subplot(gs[3])
    for m, c in zip(all_metrics, COLORS):
        eq = m["equity"]
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        ax4.fill_between(eq.index, dd, 0, alpha=0.3, color=c)
        ax4.plot(eq.index, dd, color=c, lw=1, label=m["label"])
    ax4.axhline(0, color=WHITE, lw=0.5)
    ax4.legend(fontsize=8, facecolor=PANEL, labelcolor=WHITE)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"{x:.0f}%"))
    style_ax(ax4, "Drawdown by Model (%)")

    fig.suptitle(
        "ML Alpha — Model Results (Layer 2)  |  Walk-Forward Out-of-Sample  |  2015-2024",
        color=WHITE, fontsize=13, fontweight="bold", y=0.999)

    plt.savefig("ml_model_results.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    print("\n[Layer 2] Chart saved -> ml_model_results.png")
    plt.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("[Layer 2] Loading data...")
    df     = pd.read_csv("ml_features.csv",
                         index_col=["date","ticker"], parse_dates=["date"])
    prices = pd.read_csv("prices_monthly.csv",
                         index_col=0, parse_dates=True)
    print(f"  Features: {df.shape}")

    # Try XGBoost, fall back to GradientBoosting
    try:
        from xgboost import XGBRegressor
        gb = XGBRegressor(n_estimators=100, max_depth=4,
                          learning_rate=0.05, subsample=0.8,
                          random_state=42, verbosity=0)
        gb_name = "XGBoost"
    except ImportError:
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                        learning_rate=0.05, subsample=0.8,
                                        random_state=42)
        gb_name = "Gradient Boost"

    models = [
        ("Ridge Regression",
         Ridge(alpha=1.0)),
        ("Random Forest",
         RandomForestRegressor(n_estimators=100, max_depth=5,
                               min_samples_leaf=20,
                               random_state=42, n_jobs=-1)),
        (gb_name, gb),
    ]

    # Walk-forward predictions
    all_pred_dfs = {}
    all_ics      = {}

    for name, model in models:
        print(f"\n{'─'*50}")
        pred_df, ics = walk_forward_predict(df, model,
                                             train_years=4, name=name)
        all_pred_dfs[name] = pred_df
        all_ics[name]      = ics

    # Simulate portfolios
    print("\n[Layer 2] Simulating portfolios...")
    all_metrics = []
    benchmark   = prices.pct_change().mean(axis=1)

    for name, pred_df in all_pred_dfs.items():
        if pred_df.empty:
            continue
        port_rets = simulate_portfolio(pred_df, prices)
        m = compute_metrics(port_rets, label=name)
        if m:
            all_metrics.append(m)

    # Summary
    print(f"\n{'='*58}")
    print(f"  MODEL COMPARISON — Out-of-Sample Walk-Forward")
    print(f"{'='*58}")
    print(f"  {'Model':<22} {'Ann Ret':>9} {'Sharpe':>8} "
          f"{'Max DD':>9} {'Calmar':>8}")
    print(f"  {'-'*54}")
    for m in all_metrics:
        print(f"  {m['label']:<22} {m['ann_ret']:>8.1f}% "
              f"{m['sharpe']:>8.3f} "
              f"{m['max_dd']:>8.1f}% "
              f"{m['calmar']:>8.3f}")

    # Align benchmark
    if all_metrics:
        bench_aligned = benchmark.reindex(all_metrics[0]["equity"].index)
    else:
        bench_aligned = None

    plot_results(all_metrics, all_ics, bench_aligned)

    # Save predictions
    for name, pred_df in all_pred_dfs.items():
        fname = f"predictions_{name.lower().replace(' ','_')}.csv"
        pred_df.to_csv(fname, index=False)
    print("[Done] Predictions saved -> predictions_*.csv")
