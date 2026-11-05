"""Regression models and evaluation for the NSE closing-price study.

Each model is a per-stock OLS regression of the day's closing price, fitted
on one year of training days and evaluated on the following month. For every
test day the model gives a point forecast and a 95% prediction interval.

Models
------
Simple (original)      close_t ~ prev_close_t
Same-day (original)    close_t ~ prev_close_t + low_t + trades_t
Lagged (fixed)         close_t ~ prev_close_t + low_(t-1) + trades_(t-1)
At-open (fixed)        close_t ~ prev_close_t + open_t
Random walk            close_t = prev_close_t

The same-day model uses the day's low and number of trades, which are only
known once the market closes, so it is not a real forecast (look-ahead bias).
The "fixed" models use only information available before the close.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

MODELS = {
    "Simple (original)": ["prev_close"],
    "Same-day (original)": ["prev_close", "low", "trades"],
    "Lagged (fixed)": ["prev_close", "low_lag1", "trades_lag1"],
    "At-open (fixed)": ["prev_close", "open"],
}
USES_FUTURE_INFO = {"Same-day (original)"}


def vif_table(df: pd.DataFrame, columns) -> pd.Series:
    """Variance inflation factors; values above ~10 signal severe multicollinearity."""
    X = sm.add_constant(df[columns].dropna(), has_constant="add")
    return pd.Series([variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])],
                     index=columns, name="VIF")


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features, alpha: float = 0.05):
    """Fit OLS on one stock's training days; forecast + prediction interval for its test days."""
    tr = train.dropna(subset=features + ["close"])
    te = test.dropna(subset=features + ["close"])
    res = sm.OLS(tr["close"], sm.add_constant(tr[features], has_constant="add")).fit()
    pred = res.get_prediction(sm.add_constant(te[features], has_constant="add")).summary_frame(alpha=alpha)
    out = pd.DataFrame({
        "date": te["date"].values, "actual": te["close"].values,
        "forecast": pred["mean"].values, "lower": pred["obs_ci_lower"].values,
        "upper": pred["obs_ci_upper"].values,
    })
    return out, res


def random_walk(train: pd.DataFrame, test: pd.DataFrame, alpha: float = 0.05):
    """Forecast = previous close; interval from the spread of daily changes in training."""
    sd = (train["close"] - train["prev_close"]).std(ddof=1)
    z = stats.norm.ppf(1 - alpha / 2)
    te = test.dropna(subset=["close", "prev_close"])
    f = te["prev_close"].values
    return pd.DataFrame({"date": te["date"].values, "actual": te["close"].values,
                         "forecast": f, "lower": f - z * sd, "upper": f + z * sd})


def score(fc: pd.DataFrame) -> dict:
    err = fc["actual"] - fc["forecast"]
    inside = (fc["actual"] >= fc["lower"]) & (fc["actual"] <= fc["upper"])
    return {
        "MAPE_%": 100 * np.mean(np.abs(err / fc["actual"])),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "coverage_%": 100 * inside.mean(),
        "interval_width_%": 100 * np.mean((fc["upper"] - fc["lower"]) / fc["actual"]),
        "n_test_days": len(fc),
    }


def diebold_mariano(e1, e2) -> float:
    """Two-sided p-value for equal squared-error accuracy (h=1, Harvey correction)."""
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    n = len(d)
    if n < 3 or d.var(ddof=1) == 0:
        return np.nan
    dm = d.mean() / np.sqrt(d.var(ddof=1) / n) * np.sqrt((n - 1) / n)
    return 2 * stats.t.sf(abs(dm), df=n - 1)


def run_all(train: pd.DataFrame, test: pd.DataFrame, min_train_days: int = 60):
    """Fit every model for every stock. Returns per-stock scores and all forecasts."""
    rows, forecasts = [], {}
    for sym, tr in train.groupby("symbol"):
        te = test[test["symbol"] == sym]
        if len(tr) < min_train_days or te.empty:
            continue
        rw = random_walk(tr, te)
        forecasts[(sym, "Random walk")] = rw
        rows.append({"symbol": sym, "model": "Random walk", "train_days": len(tr), **score(rw)})
        for name, feats in MODELS.items():
            fc, res = fit_predict(tr, te, feats)
            forecasts[(sym, name)] = fc
            merged = fc.merge(rw[["date", "forecast"]], on="date", suffixes=("", "_rw"))
            rows.append({
                "symbol": sym, "model": name, "train_days": len(tr), **score(fc),
                "R2_train": res.rsquared,
                "DM_p_vs_RW": diebold_mariano(merged["actual"] - merged["forecast"],
                                              merged["actual"] - merged["forecast_rw"]),
                "beats_RW": score(fc)["RMSE"] < score(rw)["RMSE"],
            })
    return pd.DataFrame(rows), forecasts


def summarise(scores: pd.DataFrame, good_coverage: float = 90.0) -> pd.DataFrame:
    """One row per model, aggregated over stocks."""
    sc = scores.copy()
    sc["beats"] = sc["beats_RW"].astype("float")              # NaN for the random walk itself
    sc["sig_better"] = ((sc["DM_p_vs_RW"] < 0.05) & (sc["beats"] == 1)).astype(float)
    sc.loc[sc["model"] == "Random walk", "sig_better"] = np.nan
    sc["good"] = (sc["coverage_%"] >= good_coverage).astype(float)
    g = sc.groupby("model", sort=False)
    out = pd.DataFrame({
        "stocks": g.size(),
        "median_MAPE_%": g["MAPE_%"].median(),
        "mean_coverage_%": g["coverage_%"].mean(),
        f"stocks_with_coverage_>={good_coverage:.0f}%": g["good"].sum().astype(int),
        "median_interval_width_%": g["interval_width_%"].median(),
        "stocks_beating_RW": g["beats"].sum(min_count=1),
        "stocks_sig_better_than_RW": g["sig_better"].sum(min_count=1),
    })
    out["uses_future_info"] = [m in USES_FUTURE_INFO for m in out.index]
    return out


def simulate_placebo(n_stocks: int = 30, seed: int = 0) -> pd.DataFrame:
    """Fake stocks whose prices are pure random walks: nothing is predictable.

    Any model that 'beats' the random walk on this data is getting its edge
    from information it shouldn't have (here: the same day's low).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-03-02", "2021-04-30")
    rows = []
    for k in range(n_stocks):
        prev = rng.uniform(50, 3000)
        prev_low, prev_trades = np.nan, np.nan
        for d in dates:
            o = prev * np.exp(rng.normal(0, 0.01))
            c = o * np.exp(rng.normal(0, 0.018))
            low = min(o, c) * np.exp(-abs(rng.normal(0, 0.008)))
            trades = rng.lognormal(10, 0.5)  # unrelated to price
            rows.append({"symbol": f"FAKE{k:02d}", "date": d, "prev_close": prev, "open": o,
                         "low": low, "close": c, "trades": trades,
                         "low_lag1": prev_low, "trades_lag1": prev_trades})
            prev, prev_low, prev_trades = c, low, trades
    return pd.DataFrame(rows)
