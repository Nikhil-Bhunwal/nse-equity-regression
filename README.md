# Predicting NSE Closing Prices with Regression, and a Look-Ahead Bias Check
**30 NSE stocks · per-stock OLS with 95% prediction intervals · random-walk benchmark · placebo test**

This project rebuilds my 2021 term project at IIT Kharagpur (supervisor: Dr. Debjani Chakraborty). The original found that a multiple regression on *previous close, day low and number of trades* predicted closing prices within a 95% band for 29/30 stocks, against 14/30 for a simple regression on previous close.

The rebuild asks a model-validation question: **the day's low and number of trades are only known after the close, so is that model really forecasting?**

> 📊 **Results:** *pending the run on NSE data*

## What's inside
| Step | Details |
|---|---|
| **Data** | NSE *Security-wise Price volume & Deliverable position* report, Mar 2020 – Apr 2021: prices, volume, turnover, number of trades, deliverable quantity |
| **Variable selection** | Variance inflation factors for the 11 candidate variables, explaining why most were dropped |
| **Models** (one per stock) | Simple: `close ~ prev_close` · Same-day (original): `+ low_t + trades_t` · Lagged (fixed): `+ low_(t-1) + trades_(t-1)` · At-open (fixed): `+ open_t` · Random walk |
| **Evaluation** | Train 17 Mar 2020 – 16 Mar 2021, test 17 Mar – 18 Apr 2021. MAPE, 95% interval coverage, Diebold–Mariano test vs random walk |
| **Placebo test** | The same models run on simulated random-walk stocks where nothing is predictable. A model that still "wins" there is using future information |

## Project structure
```
nse-equity-regression/
├── data/raw/                       # NSE download (nse_prices.csv)
├── notebooks/
│   └── nse_equity_regression.ipynb # the full analysis, start here
├── results/                        # summary tables and figures
├── src/
│   ├── download_data.py            # fetches the 30 stocks from NSE (nselib)
│   ├── data.py                     # cleaning, lagged features, train/test split
│   └── models.py                   # VIF, OLS models, random walk, scoring, placebo
└── requirements.txt
```

## How to run
```bash
git clone https://github.com/Nikhil-Bhunwal/nse-equity-regression.git
cd nse-equity-regression
pip install -r requirements.txt
python src/download_data.py
jupyter notebook notebooks/nse_equity_regression.ipynb
```

## Author
**Nikhil Bhunwal** · M.Sc. Mathematics & Computing, IIT Kharagpur · [LinkedIn](https://linkedin.com/in/nikhil-bhunwal-2963a0165) · [GitHub](https://github.com/Nikhil-Bhunwal)
