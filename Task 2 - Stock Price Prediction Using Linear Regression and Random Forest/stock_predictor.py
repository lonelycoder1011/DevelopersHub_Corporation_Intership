"""
╔══════════════════════════════════════════════════════════════╗
║         STOCK PRICE PREDICTOR  —  Assignment 2               ║
║  Predict Next Day's Closing Price using Machine Learning      ║
╠══════════════════════════════════════════════════════════════╣
║  Models  : Linear Regression  +  Random Forest Regressor     ║
║  Data    : Yahoo Finance via yfinance  (cached to CSV)        ║
║  Plots   : EDA · Train/Val Loss · Actual vs Predicted         ║
║            Scatter · Feature Importance · Error Distribution  ║
║            Summary Dashboard                                  ║
╚══════════════════════════════════════════════════════════════╝

  ┌─ HOW TO CHANGE THE STOCK ──────────────────────────────────┐
  │  Edit the CONFIG block below — just change TICKER.         │
  │  Examples:                                                  │
  │    "AAPL"  → Apple        "TSLA"  → Tesla                  │
  │    "MSFT"  → Microsoft    "GOOGL" → Google                 │
  │    "AMZN"  → Amazon       "META"  → Meta                   │
  │    "NVDA"  → NVIDIA       "NFLX"  → Netflix                │
  └────────────────────────────────────────────────────────────┘
"""

# ┌─────────────────────────────────────────────────────────┐
# │  CONFIG  ← only section you need to edit                │
# └─────────────────────────────────────────────────────────┘
TICKER           = "NVDA"   # Any valid Yahoo Finance tickerAAPL
PERIOD           = "2y"     # "1y" | "2y" | "5y"  (2y ≈ 504 rows)
VALIDATION_SPLIT = 0.15     # 15% of training data used for validation
TEST_SPLIT       = 0.20     # 20% of full data held out for final test
RF_N_TREES       = 100      # keep ≤150 for laptop comfort
RF_MAX_DEPTH     = 8
RANDOM_SEED      = 42
OUTPUT_DIR       = "outputs"  # plots & CSV go here
DATA_DIR         = "data"     # fetched CSV data is cached here
# ──────────────────────────────────────────────────────────

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

import yfinance as yf
from sklearn.linear_model   import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing   import StandardScaler

warnings.filterwarnings("ignore")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

# ── Colour palette ────────────────────────────────────────
C = {
    "actual" : "#1565C0",
    "lr"     : "#E64A19",
    "rf"     : "#2E7D32",
    "val"    : "#6A1B9A",
    "train"  : "#F57F17",
    "grid"   : "#E0E0E0",
    "bg"     : "#FAFAFA",
}

# ══════════════════════════════════════════════════════════
#  1.  DATA — Load & Cache
# ══════════════════════════════════════════════════════════
def load_data(ticker: str, period: str) -> pd.DataFrame:
    """
    Fetches data from Yahoo Finance and saves it as a CSV file
    in the  data/  folder so that re-runs work without internet.
    If the cache already exists it is loaded directly.

    Cache location:  data/<TICKER>_<PERIOD>.csv
    """
    cache_path = os.path.join(DATA_DIR, f"{ticker}_{period}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if not df.empty:
            print(f"  ✔  Loaded from cache  : {cache_path}")
            print(f"  ✔  Rows               : {len(df)}")
            print(f"  ✔  Date range         : {df.index[0].date()} → {df.index[-1].date()}")
            return df

    print(f"  ↓  Fetching {ticker} from Yahoo Finance …")
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(
            f"No data returned for '{ticker}'. "
            "Check the symbol and your internet connection."
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.dropna(inplace=True)
    df.to_csv(cache_path)

    print(f"  ✔  Rows fetched       : {len(df)}")
    print(f"  ✔  Date range         : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  ✔  Saved to cache     : {cache_path}")
    print(f"  ✔  Columns            : {list(df.columns)}")
    return df


# ══════════════════════════════════════════════════════════
#  2.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════
FEATURES = [
    "Open", "High", "Low", "Volume",
    "Range", "Prev_Close",
    "SMA_5", "SMA_10",
    "Volatility_5",
    "Return_1d", "Return_5d",
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derived features:
      Range        = High − Low          (intraday spread)
      Prev_Close   = Close(t-1)          (momentum anchor)
      SMA_5/10     = rolling mean close  (trend signals)
      Volatility_5 = 5-day rolling std   (risk proxy)
      Return_1d/5d = short-term momentum
      Target       = Close(t+1)          ← what we predict
    """
    d = df.copy()
    d["Range"]        = d["High"]  - d["Low"]
    d["Prev_Close"]   = d["Close"].shift(1)
    d["SMA_5"]        = d["Close"].rolling(5).mean()
    d["SMA_10"]       = d["Close"].rolling(10).mean()
    d["Volatility_5"] = d["Close"].rolling(5).std()
    d["Return_1d"]    = d["Close"].pct_change(1) * 100
    d["Return_5d"]    = d["Close"].pct_change(5) * 100
    d["Target"]       = d["Close"].shift(-1)    # NEXT day's close
    d.dropna(inplace=True)
    return d


# ══════════════════════════════════════════════════════════
#  3.  CHRONOLOGICAL  TRAIN / VAL / TEST  SPLIT
# ══════════════════════════════════════════════════════════
def split_data(data: pd.DataFrame, val_ratio: float, test_ratio: float):
    """
    Strict chronological split — absolutely no data leakage.

      ├──── TRAIN ────────────────┤── VAL ──┤── TEST ──┤
      0                        val_idx  test_idx       n
    """
    n        = len(data)
    test_idx = int(n * (1 - test_ratio))
    val_idx  = int(test_idx * (1 - val_ratio))

    train = data.iloc[:val_idx]
    val   = data.iloc[val_idx:test_idx]
    test  = data.iloc[test_idx:]

    print(f"\n  Split → Train: {len(train)}  |  Val: {len(val)}  |  Test: {len(test)}")
    print(f"  Train : {train.index[0].date()} → {train.index[-1].date()}")
    print(f"  Val   : {val.index[0].date()} → {val.index[-1].date()}")
    print(f"  Test  : {test.index[0].date()} → {test.index[-1].date()}")

    return (
        train[FEATURES], train["Target"],
        val[FEATURES],   val["Target"],
        test[FEATURES],  test["Target"],
        test.index,
    )


# ══════════════════════════════════════════════════════════
#  4.  TRAIN WITH LOSS TRACKING
# ══════════════════════════════════════════════════════════
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(
        np.asarray(y_true), np.asarray(y_pred))))


def train_linear_regression(X_train, y_train, X_val, y_val, scaler):
    """
    Linear Regression doesn't train iteratively, so we build a
    learning curve by fitting on progressively larger fractions
    of the training data. This creates a genuine train/val loss curve.
    """
    X_tr_sc = scaler.transform(X_train)
    X_vl_sc = scaler.transform(X_val)

    checkpoints = np.linspace(0.05, 1.0, 25)
    tr_loss, vl_loss = [], []

    for frac in checkpoints:
        n = max(15, int(len(X_tr_sc) * frac))
        m = LinearRegression()
        m.fit(X_tr_sc[:n], y_train.iloc[:n])
        tr_loss.append(rmse(y_train.iloc[:n], m.predict(X_tr_sc[:n])))
        vl_loss.append(rmse(y_val,            m.predict(X_vl_sc)))

    final = LinearRegression()
    final.fit(X_tr_sc, y_train)
    return final, tr_loss, vl_loss, list(checkpoints * 100)


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Random Forest loss curve via warm_start — we grow the forest
    5 trees at a time and record RMSE at each step.
    This shows genuine training dynamics as complexity increases.
    """
    rf = RandomForestRegressor(
        n_estimators     = 5,
        max_depth        = RF_MAX_DEPTH,
        min_samples_leaf = 5,
        warm_start       = True,
        random_state     = RANDOM_SEED,
        n_jobs           = 2,
    )

    steps = list(range(5, RF_N_TREES + 1, 5))
    tr_loss, vl_loss = [], []

    for n in steps:
        rf.set_params(n_estimators=n)
        rf.fit(X_train, y_train)
        tr_loss.append(rmse(y_train, rf.predict(X_train)))
        vl_loss.append(rmse(y_val,   rf.predict(X_val)))

    return rf, tr_loss, vl_loss, steps


# ══════════════════════════════════════════════════════════
#  5.  EVALUATE
# ══════════════════════════════════════════════════════════
def evaluate(name: str, y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae    = mean_absolute_error(y_true, y_pred)
    _rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2     = r2_score(y_true, y_pred)
    mape   = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)

    print(f"\n  [{name}]")
    print(f"    MAE   : ${mae:.4f}")
    print(f"    RMSE  : ${_rmse:.4f}")
    print(f"    R²    : {r2:.4f}")
    print(f"    MAPE  : {mape:.2f}%")
    return {"Model": name, "MAE": round(mae, 4), "RMSE": round(_rmse, 4),
            "R2": round(r2, 4), "MAPE%": round(mape, 2)}


# ══════════════════════════════════════════════════════════
#  6.  PLOTTING HELPERS
# ══════════════════════════════════════════════════════════
def _style_ax(ax, title="", xlabel="", ylabel="", legend=True):
    ax.set_facecolor(C["bg"])
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(color=C["grid"], linewidth=0.8, linestyle="--", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    if legend: ax.legend(fontsize=8, framealpha=0.85, edgecolor=C["grid"])
    ax.tick_params(labelsize=8)


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔  Saved → {path}")


# ══════════════════════════════════════════════════════════
#  PLOT 1 — EDA
# ══════════════════════════════════════════════════════════
def plot_eda(df: pd.DataFrame, ticker: str):
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    fig.suptitle(f"{ticker}  —  Exploratory Data Analysis",
                 fontsize=16, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2])

    close = df["Close"].squeeze()

    # Price history
    ax1.fill_between(df.index, close.values, float(close.min()), alpha=0.12, color=C["actual"])
    ax1.plot(df.index, close.values, color=C["actual"], lw=1.6, label="Close")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30)
    _style_ax(ax1, "Closing Price History", ylabel="Price (USD)")

    # Volume
    vol = df["Volume"].squeeze()
    ax2.bar(df.index, vol.values, color="#7B1FA2", alpha=0.55, width=1.5)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30)
    _style_ax(ax2, "Trading Volume", ylabel="Volume (M shares)", legend=False)

    # Moving averages
    sma5  = close.rolling(5).mean()
    sma20 = close.rolling(20).mean()
    ax3.plot(df.index, close.values, lw=1.0, color="#B0BEC5", label="Close", alpha=0.9)
    ax3.plot(df.index, sma5.values,  lw=1.7, color="#FF6F00", label="SMA-5")
    ax3.plot(df.index, sma20.values, lw=1.7, color="#C62828", label="SMA-20")
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30)
    _style_ax(ax3, "Price with Moving Averages (SMA-5 & SMA-20)", ylabel="Price (USD)")

    # Daily returns histogram
    returns = (close.pct_change().dropna() * 100).values
    mean_r, std_r = returns.mean(), returns.std()
    n_hist, bins, patches = ax4.hist(returns, bins=40, edgecolor="white", alpha=0.85)
    for patch, b in zip(patches, bins):
        patch.set_facecolor("#C62828" if b < 0 else "#2E7D32")
    ax4.axvline(mean_r, color="black", lw=1.5, ls="--", label=f"Mean: {mean_r:.2f}%")
    ax4.axvline(mean_r - std_r, color="grey", lw=1.0, ls=":", label="±1σ")
    ax4.axvline(mean_r + std_r, color="grey", lw=1.0, ls=":")
    _style_ax(ax4, "Daily Return Distribution", xlabel="Return (%)", ylabel="Count")

    _save(fig, os.path.join(OUTPUT_DIR, f"{ticker}_1_eda.png"))


# ══════════════════════════════════════════════════════════
#  PLOT 2 — Train / Validation Loss Curves
# ══════════════════════════════════════════════════════════
def plot_loss_curves(lr_tr, lr_vl, lr_x,
                     rf_tr, rf_vl, rf_x,
                     ticker: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    fig.suptitle(f"{ticker}  —  Train vs Validation Loss  (RMSE in $)",
                 fontsize=14, fontweight="bold", y=1.01)

    configs = [
        (axes[0], lr_tr, lr_vl, lr_x,
         "Linear Regression — Learning Curve\n(trained on increasing data fraction)",
         "Training Data Used (%)", C["lr"]),
        (axes[1], rf_tr, rf_vl, rf_x,
         "Random Forest — Staged Loss\n(RMSE vs number of trees)",
         "Number of Trees", C["rf"]),
    ]

    for ax, tr, vl, xs, title, xlabel, color in configs:
        ax.set_facecolor(C["bg"])

        # Shaded region between curves
        ax.fill_between(xs, tr, vl,
                        where=[v >= t for v, t in zip(vl, tr)],
                        alpha=0.12, color=C["val"],
                        label="Generalisation Gap")
        ax.fill_between(xs, tr, vl,
                        where=[v < t for v, t in zip(vl, tr)],
                        alpha=0.08, color=C["train"])

        ax.plot(xs, tr, color=C["train"], lw=2.2, marker="o",
                markersize=4, label="Train Loss (RMSE)", zorder=3)
        ax.plot(xs, vl, color=C["val"],   lw=2.2, marker="s",
                markersize=4, linestyle="--", label="Val Loss (RMSE)", zorder=3)

        # Annotate best validation point
        min_i   = int(np.argmin(vl))
        min_val = vl[min_i]
        ax.annotate(
            f"Best val\n${min_val:.2f}",
            xy=(xs[min_i], min_val),
            xytext=(xs[min_i] + (max(xs) - min(xs)) * 0.08, min_val * 1.04),
            fontsize=7.5, ha="left",
            arrowprops=dict(arrowstyle="->", color="#333", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", lw=0.8),
        )

        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("RMSE ($)", fontsize=9)
        ax.grid(color=C["grid"], linewidth=0.8, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, framealpha=0.85)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    _save(fig, os.path.join(OUTPUT_DIR, f"{ticker}_2_loss_curves.png"))


# ══════════════════════════════════════════════════════════
#  PLOT 3 — Actual vs Predicted (time series)
# ══════════════════════════════════════════════════════════
def plot_actual_vs_predicted(dates, y_true, pred_lr, pred_rf, ticker: str):
    fig, axes = plt.subplots(2, 1, figsize=(15, 9),
                              sharex=True, facecolor="white")
    fig.suptitle(f"{ticker}  —  Actual vs Predicted  (Test Set)",
                 fontsize=15, fontweight="bold", y=0.99)

    for ax, pred, color, name in [
        (axes[0], pred_lr, C["lr"], "Linear Regression"),
        (axes[1], pred_rf, C["rf"], "Random Forest"),
    ]:
        ax.plot(dates, y_true, color=C["actual"], lw=1.9,
                label="Actual Close", zorder=3)
        ax.plot(dates, pred, color=color, lw=1.5,
                linestyle="--", label=f"{name} Prediction", zorder=4)

        ax.fill_between(dates, y_true, pred,
                        where=(pred > y_true),
                        alpha=0.10, color="#C62828", label="Over-prediction")
        ax.fill_between(dates, y_true, pred,
                        where=(pred <= y_true),
                        alpha=0.10, color="#1565C0", label="Under-prediction")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)
        _style_ax(ax, name, ylabel="Price (USD)")

    plt.tight_layout()
    _save(fig, os.path.join(OUTPUT_DIR, f"{ticker}_3_actual_vs_predicted.png"))


# ══════════════════════════════════════════════════════════
#  PLOT 4 — Scatter: Predicted vs Actual
# ══════════════════════════════════════════════════════════
def plot_scatter(y_true, pred_lr, pred_rf, ticker: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    fig.suptitle(f"{ticker}  —  Predicted vs Actual  (Scatter)",
                 fontsize=13, fontweight="bold")

    for ax, pred, color, name in [
        (axes[0], pred_lr, C["lr"], "Linear Regression"),
        (axes[1], pred_rf, C["rf"], "Random Forest"),
    ]:
        mn = min(float(y_true.min()), float(pred.min())) * 0.98
        mx = max(float(y_true.max()), float(pred.max())) * 1.02
        ax.scatter(y_true, pred, color=color, alpha=0.5,
                   s=18, edgecolors="white", lw=0.4)
        ax.plot([mn, mx], [mn, mx], "k--", lw=1.2, label="Perfect fit")
        ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
        r2 = r2_score(y_true, pred)
        _style_ax(ax, f"{name}  |  R² = {r2:.4f}",
                  xlabel="Actual Close ($)", ylabel="Predicted Close ($)")

    plt.tight_layout()
    _save(fig, os.path.join(OUTPUT_DIR, f"{ticker}_4_scatter.png"))


# ══════════════════════════════════════════════════════════
#  PLOT 5 — Feature Importance
# ══════════════════════════════════════════════════════════
def plot_feature_importance(rf_model, ticker: str):
    imp    = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values()
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(imp)))

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    bars = ax.barh(imp.index, imp.values, color=colors,
                   edgecolor="white", height=0.65)
    for bar, val in zip(bars, imp.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8.5)
    _style_ax(ax, f"{ticker}  —  Random Forest Feature Importance",
              xlabel="Importance Score", legend=False)
    ax.set_xlim(0, float(imp.max()) * 1.18)

    plt.tight_layout()
    _save(fig, os.path.join(OUTPUT_DIR, f"{ticker}_5_feature_importance.png"))


# ══════════════════════════════════════════════════════════
#  PLOT 6 — Error Distribution
# ══════════════════════════════════════════════════════════
def plot_error_distribution(y_true, pred_lr, pred_rf, ticker: str):
    err_lr = np.asarray(y_true) - np.asarray(pred_lr)
    err_rf = np.asarray(y_true) - np.asarray(pred_rf)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")
    fig.suptitle(f"{ticker}  —  Prediction Error Distribution  (Test Set)",
                 fontsize=13, fontweight="bold")

    for ax, err, name, color in [
        (axes[0], err_lr, "Linear Regression", C["lr"]),
        (axes[1], err_rf, "Random Forest",     C["rf"]),
    ]:
        ax.hist(err, bins=35, color=color, alpha=0.75, edgecolor="white")
        ax.axvline(0,          color="black", lw=1.8, ls="--", label="Zero error")
        ax.axvline(err.mean(), color="gold",  lw=1.5, ls="-",
                   label=f"Mean: ${err.mean():.2f}")
        ax.text(0.97, 0.96,
                f"Std  : ${err.std():.2f}\nMin  : ${err.min():.2f}\nMax : ${err.max():.2f}",
                transform=ax.transAxes, va="top", ha="right",
                fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85))
        _style_ax(ax, name, xlabel="Error  (Actual − Predicted)  $", ylabel="Frequency")

    plt.tight_layout()
    _save(fig, os.path.join(OUTPUT_DIR, f"{ticker}_6_error_distribution.png"))


# ══════════════════════════════════════════════════════════
#  PLOT 7 — Summary Dashboard
# ══════════════════════════════════════════════════════════
def plot_dashboard(metrics: list, ticker: str,
                   last_close, next_lr: float, next_rf: float):
    fig = plt.figure(figsize=(11, 4.2), facecolor="#1A237E")
    fig.suptitle(f"  {ticker}  —  Model Summary Dashboard",
                 fontsize=15, fontweight="bold", color="white", x=0.5, y=0.96)

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    cards = [
        (0.04, 0.08, metrics[0]["Model"], metrics[0], C["lr"]),
        (0.37, 0.08, metrics[1]["Model"], metrics[1], C["rf"]),
        (0.70, 0.08, "Next-Day Forecast", None,       "#1565C0"),
    ]

    for (x, y, title, m, color) in cards:
        rect = FancyBboxPatch(
            (x, y), 0.28, 0.76,
            boxstyle="round,pad=0.02",
            linewidth=2, edgecolor=color,
            facecolor="white", alpha=0.93,
            transform=ax.transAxes, zorder=2,
        )
        ax.add_patch(rect)
        ax.text(x + 0.14, y + 0.74, title,
                ha="center", va="top", fontsize=9.5,
                fontweight="bold", color=color, transform=ax.transAxes)

        if m:
            rows = [("MAE",  f"${m['MAE']}"),
                    ("RMSE", f"${m['RMSE']}"),
                    ("R²",   f"{m['R2']}"),
                    ("MAPE", f"{m['MAPE%']}%")]
            for i, (lbl, val) in enumerate(rows):
                yy = y + 0.55 - i * 0.14
                ax.text(x + 0.05, yy, lbl,  ha="left",  va="center",
                        fontsize=9,   color="#555", transform=ax.transAxes)
                ax.text(x + 0.23, yy, val,  ha="right", va="center",
                        fontsize=9.5, fontweight="bold", color="#111",
                        transform=ax.transAxes)
        else:
            ax.text(x + 0.14, y + 0.57, "Last Close",
                    ha="center", va="center", fontsize=8.5,
                    color="#555", transform=ax.transAxes)
            ax.text(x + 0.14, y + 0.43, f"${float(last_close):.2f}",
                    ha="center", va="center", fontsize=13,
                    fontweight="bold", color="#111", transform=ax.transAxes)
            ax.text(x + 0.14, y + 0.28, f"Lin. Reg. → ${next_lr:.2f}",
                    ha="center", va="center", fontsize=9.5,
                    color=C["lr"], transform=ax.transAxes)
            ax.text(x + 0.14, y + 0.14, f"Ran. For. → ${next_rf:.2f}",
                    ha="center", va="center", fontsize=9.5,
                    color=C["rf"], transform=ax.transAxes)

    fig.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_7_dashboard.png"),
                dpi=150, bbox_inches="tight", facecolor="#1A237E")
    plt.close(fig)
    print(f"  ✔  Saved → {os.path.join(OUTPUT_DIR, f'{ticker}_7_dashboard.png')}")


# ══════════════════════════════════════════════════════════
#  SAVE METRICS CSV
# ══════════════════════════════════════════════════════════
def save_metrics_csv(results: list, ticker: str):
    df_res = pd.DataFrame(results)
    path   = os.path.join(OUTPUT_DIR, f"{ticker}_metrics.csv")
    df_res.to_csv(path, index=False)
    print(f"  ✔  Metrics CSV → {path}")
    print(f"\n{'─'*52}")
    print(df_res.to_string(index=False))
    print(f"{'─'*52}")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
def main():
    print("\n" + "█"*58)
    print("  STOCK PRICE PREDICTOR  —  Assignment 2")
    print("█"*58)

    # ─ 1. Load ────────────────────────────────────────────
    sep = f"\n{'─'*58}"
    print(f"{sep}\n  STEP 1 : Load Data{sep}")
    df_raw = load_data(TICKER, PERIOD)

    # ─ 2. EDA ─────────────────────────────────────────────
    print(f"{sep}\n  STEP 2 : Exploratory Data Analysis{sep}")
    plot_eda(df_raw, TICKER)

    # ─ 3. Feature engineering ─────────────────────────────
    print(f"{sep}\n  STEP 3 : Feature Engineering{sep}")
    data = build_features(df_raw)
    print(f"  Features : {FEATURES}")
    print(f"  Total rows after engineering : {len(data)}")

    # ─ 4. Split ───────────────────────────────────────────
    print(f"{sep}\n  STEP 4 : Train / Val / Test Split{sep}")
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     test_dates) = split_data(data, VALIDATION_SPLIT, TEST_SPLIT)

    # ─ 5. Scale (fit ONLY on training data) ───────────────
    scaler = StandardScaler()
    scaler.fit(X_train)

    # ─ 6. Train ───────────────────────────────────────────
    print(f"{sep}\n  STEP 5 : Train Models + Track Loss{sep}")
    print("  Training Linear Regression …")
    lr_model, lr_tr, lr_vl, lr_x = train_linear_regression(
        X_train, y_train, X_val, y_val, scaler)

    print("  Training Random Forest …")
    rf_model, rf_tr, rf_vl, rf_x = train_random_forest(
        X_train, y_train, X_val, y_val)
    print("  ✔  Both models trained.")

    # ─ 7. Predict ─────────────────────────────────────────
    pred_lr = lr_model.predict(scaler.transform(X_test))
    pred_rf = rf_model.predict(X_test)

    # ─ 8. Evaluate ────────────────────────────────────────
    print(f"{sep}\n  STEP 6 : Evaluate on Test Set{sep}")
    results = [
        evaluate("Linear Regression", y_test, pred_lr),
        evaluate("Random Forest",     y_test, pred_rf),
    ]
    save_metrics_csv(results, TICKER)

    # ─ 9. All plots ───────────────────────────────────────
    print(f"{sep}\n  STEP 7 : Generate Plots{sep}")
    plot_loss_curves(lr_tr, lr_vl, lr_x,
                     rf_tr, rf_vl, rf_x, TICKER)
    plot_actual_vs_predicted(test_dates, y_test.values, pred_lr, pred_rf, TICKER)
    plot_scatter(y_test.values, pred_lr, pred_rf, TICKER)
    plot_feature_importance(rf_model, TICKER)
    plot_error_distribution(y_test.values, pred_lr, pred_rf, TICKER)

    # ─ 10. Next-day forecast ──────────────────────────────
    last_row   = data[FEATURES].iloc[[-1]]
    next_lr    = float(lr_model.predict(scaler.transform(last_row))[0])
    next_rf    = float(rf_model.predict(last_row)[0])
    last_close = df_raw["Close"].iloc[-1]

    plot_dashboard(results, TICKER, last_close, next_lr, next_rf)

    # ─ 11. Final print ────────────────────────────────────
    print(f"\n{'█'*58}")
    print("  NEXT-DAY CLOSING PRICE FORECAST")
    print(f"{'─'*58}")
    print(f"  Ticker            : {TICKER}")
    print(f"  Last known close  : ${float(last_close):.2f}")
    print(f"  Linear Regression : ${next_lr:.2f}")
    print(f"  Random Forest     : ${next_rf:.2f}")
    print(f"{'─'*58}")
    print(f"  Data cached at    : ./{DATA_DIR}/{TICKER}_{PERIOD}.csv")
    print(f"  All outputs in    : ./{OUTPUT_DIR}/")
    print(f"{'█'*58}\n")


if __name__ == "__main__":
    main()
