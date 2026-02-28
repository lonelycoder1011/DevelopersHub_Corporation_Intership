# 📈 Stock Price Predictor — Assignment 2

> **Predict the next trading day's closing price** using historical OHLCV data, engineered features, and two ML models: Linear Regression and Random Forest.

---

## 🗂️ Project Structure

```
stock_predictor/
│
├── stock_predictor.py      ← Main script  (run this)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
│
├── data/                   ← Auto-created — cached CSV data
│   └── AAPL_2y.csv         ← Downloaded stock data (reused on re-runs)
│
└── outputs/                ← Auto-created — all generated files
    ├── AAPL_1_eda.png
    ├── AAPL_2_loss_curves.png
    ├── AAPL_3_actual_vs_predicted.png
    ├── AAPL_4_scatter.png
    ├── AAPL_5_feature_importance.png
    ├── AAPL_6_error_distribution.png
    ├── AAPL_7_dashboard.png
    └── AAPL_metrics.csv
```

---

## 📊 Dataset Details

| Property | Value |
|---|---|
| **Source** | Yahoo Finance via `yfinance` |
| **Default Stock** | AAPL (Apple Inc.) |
| **Period** | 2 years (`2y`) |
| **Approximate Rows** | ~504 trading days |
| **Approximate Size** | ~200 KB in memory / ~60 KB on disk |
| **Features** | Open, High, Low, Volume + 7 engineered features |
| **Target** | Next day's Closing Price |

> ✅ **Why 2 years?**  504 rows (~200 KB) is deliberately lean so the entire pipeline — download, feature engineering, model training, and all 7 plots — finishes in **under 30 seconds** on an Intel i5 5th Gen / 8 GB RAM laptop.

---

## 💾 Where is the Data Stored?

After the first run, fetched data is automatically saved to:

```
data/AAPL_2y.csv    ← or whatever TICKER_PERIOD you set
```

On every subsequent run the script **reads from this cache** — no internet needed. To force a fresh download, simply delete the CSV file.

---

## 🔧 How to Change the Stock

Open `stock_predictor.py` and edit the **CONFIG block** at the top:

```python
TICKER = "AAPL"   # ← change to any valid Yahoo Finance ticker
PERIOD = "2y"     # ← "1y", "2y", or "5y"
```

**Popular tickers:**

| Company   | Ticker | Company  | Ticker |
|-----------|--------|----------|--------|
| Apple     | `AAPL` | NVIDIA   | `NVDA` |
| Tesla     | `TSLA` | Amazon   | `AMZN` |
| Microsoft | `MSFT` | Meta     | `META` |
| Google    | `GOOGL`| Netflix  | `NFLX` |

---

## 🚀 Setup & Run

```bash
# 1 — Install dependencies
pip install -r requirements.txt

# 2 — Run
python stock_predictor.py

# 3 — View outputs/  for all plots
```

---

## 🤖 Models

### Linear Regression (Baseline)
- Fits a linear mapping from features → next-day close
- Scaled with `StandardScaler`
- **Loss curve**: trained on increasing data fractions (5%→100%) to produce a learning curve

### Random Forest Regressor
- 100 trees, `max_depth=8`, `min_samples_leaf=5`, `n_jobs=2`
- Configured lightweight for laptop use
- **Loss curve**: uses `warm_start=True` to grow 5 trees at a time and record RMSE at each step

---

## 📐 Features Engineered

| Feature | Description |
|---|---|
| `Open` | Opening price |
| `High` | Daily high |
| `Low` | Daily low |
| `Volume` | Shares traded |
| `Range` | High − Low (intraday spread) |
| `Prev_Close` | Yesterday's close (momentum anchor) |
| `SMA_5` | 5-day simple moving average |
| `SMA_10` | 10-day simple moving average |
| `Volatility_5` | 5-day rolling std deviation |
| `Return_1d` | 1-day price return % |
| `Return_5d` | 5-day price return % |

---

## 📊 Output Plots

| File | Description |
|---|---|
| `_1_eda.png` | 4-panel EDA: price history, volume, SMA overlay, returns distribution |
| `_2_loss_curves.png` | **Train vs Validation loss** for both models |
| `_3_actual_vs_predicted.png` | Time-series comparison with over/under prediction shading |
| `_4_scatter.png` | Predicted vs Actual scatter with R² annotation |
| `_5_feature_importance.png` | Random Forest feature importance bar chart |
| `_6_error_distribution.png` | Error histograms with mean/std stats |
| `_7_dashboard.png` | Summary card: metrics + next-day forecast |
| `_metrics.csv` | MAE, RMSE, R², MAPE% for both models |

---

## 📉 Evaluation Metrics

| Metric | Meaning |
|---|---|
| **MAE** | Average dollar error |
| **RMSE** | Root mean squared error (penalises large errors more) |
| **R²** | Explained variance (1.0 = perfect) |
| **MAPE%** | Mean absolute percentage error |

---

## 💻 System Requirements

| Component | Minimum |
|---|---|
| CPU | Intel i5 5th Gen |
| RAM | 4 GB (8 GB recommended) |
| Python | 3.8+ |
| Internet | Only for first run (data cached after that) |

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. Stock predictions are inherently uncertain — do **not** use this for real trading decisions.
