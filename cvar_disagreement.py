# Tail risk model disagreement
# Author: Chunyang Tian
#
# Idea: I use 3 Expected Shortfall (ES) models to decide how much NVDA I can add
# to a normal portfolio. The 3 models often give different answers.
# Question: when they disagree a lot, is the next month more risky?
#
# Models: Historical Simulation (HS), Filtered Historical Simulation (FHS),
#         Student-t Monte Carlo (t-MC)

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import binomtest

# ---------------- settings ----------------
base_weights = {"SPY": 0.30, "QQQ": 0.20, "TLT": 0.20, "GLD": 0.15}  # other 15% is cash (0 return)
candidate = "NVDA"
window = 750       # use the past 750 days (about 3 years) to fit the models
step = 21          # make one decision every month
horizon = 10       # ES of the 10-day loss
alpha = 0.95       # look at the worst 5%
budget = 1.15      # ES with NVDA can be at most 15% higher than without NVDA
q_grid = np.round(np.arange(0, 0.51, 0.01), 2)   # NVDA weight from 0 to 50%
n_sim = 5000       # number of scenarios for FHS and t-MC
lam = 0.94         # EWMA decay for FHS (RiskMetrics value)
future = 20        # check the max drawdown in the next 20 days
probe_q = 0.10     # fixed 10% NVDA portfolio to measure what really happens
np.random.seed(2026)


# ---------------- 1. download data ----------------
tickers = list(base_weights.keys()) + [candidate]
price = yf.download(tickers, start="2005-01-01", auto_adjust=True)["Close"]
price = price[tickers].dropna()
ret = price.pct_change().dropna()

base_ret = 0
for t in base_weights:
    base_ret = base_ret + base_weights[t] * ret[t]
base_ret = base_ret.values
nvda_ret = ret[candidate].values
dates = ret.index
print("data from", dates[0].date(), "to", dates[-1].date(), ",", len(dates), "days")


# ---------------- 2. ES and q* ----------------
def get_es(loss):
    # ES = average of the worst 5% losses
    loss = np.sort(loss)
    k = int(np.ceil(len(loss) * (1 - alpha)))
    return loss[-k:].mean()


def find_q_star(base_paths, nvda_paths):
    # each row of base_paths / nvda_paths is one scenario of 10 daily returns
    es_list = []
    for q in q_grid:
        port = (1 - q) * base_paths + q * nvda_paths
        ten_day_ret = np.prod(1 + port, axis=1) - 1
        es_list.append(get_es(-ten_day_ret))
    es_list = np.array(es_list)
    ok = q_grid[es_list <= budget * es_list[0]]   # all q that pass the rule
    return ok.max(), es_list


def get_var_es(base_paths, nvda_paths, q):
    # the model's VaR (95%) and ES of the 10-day loss when NVDA weight is q
    port = (1 - q) * base_paths + q * nvda_paths
    loss = np.sort(-(np.prod(1 + port, axis=1) - 1))
    k = int(np.ceil(len(loss) * (1 - alpha)))
    return loss[-k], loss[-k:].mean()


# ---------------- 3. the 3 models ----------------
def hs_paths(b, c):
    # HS: every 10-day window in the past is one scenario
    b_paths, c_paths = [], []
    for i in range(len(b) - horizon + 1):
        b_paths.append(b[i:i + horizon])
        c_paths.append(c[i:i + horizon])
    return np.array(b_paths), np.array(c_paths)


def ewma_vol(x):
    # EWMA volatility for every day, and the volatility for tomorrow
    var = np.zeros(len(x))
    var[0] = x[:30].var()
    for t in range(1, len(x)):
        var[t] = lam * var[t - 1] + (1 - lam) * x[t - 1] ** 2
    var_tomorrow = lam * var[-1] + (1 - lam) * x[-1] ** 2
    return np.sqrt(var), np.sqrt(var_tomorrow)


def fhs_paths(b, c):
    # FHS: divide each return by its vol, pick random past days, times today's vol
    vol_b, today_b = ewma_vol(b)
    vol_c, today_c = ewma_vol(c)
    z_b = b / vol_b
    z_c = c / vol_c
    pick = np.random.randint(0, len(b), size=(n_sim, horizon))  # same days for both
    return z_b[pick] * today_b, z_c[pick] * today_c


def t_paths(b, c):
    # t-MC: fit mean, covariance and degree of freedom, then simulate
    data = np.column_stack([b, c])
    mean = data.mean(axis=0)
    cov = np.cov(data.T)
    kurt = pd.DataFrame(data).kurtosis().mean()      # excess kurtosis
    nu = 4 + 6 / max(kurt, 0.01)
    nu = min(max(nu, 4.5), 30)
    # multivariate t = normal / sqrt(chi2 / nu), scale it so the covariance matches the data
    L = np.linalg.cholesky(cov * (nu - 2) / nu)
    z = np.random.normal(size=(n_sim, horizon, 2)) @ L.T
    w = np.random.chisquare(nu, size=(n_sim, horizon, 1)) / nu
    sim = mean + z / np.sqrt(w)
    return sim[:, :, 0], sim[:, :, 1], nu


# ---------------- 4. hand check on one date ----------------
# the date just before the COVID crash
i = dates.get_loc(dates[dates <= "2020-02-21"][-1]) + 1
b = base_ret[i - window:i]
c = nvda_ret[i - window:i]
print("\n--- hand check, date", dates[i - 1].date(), "---")
print("model   ES(q=0)  ES(q=0.05)  ES(q=0.10)  ES(q=0.20)  limit    q*")
for name in ["HS", "FHS", "t-MC"]:
    if name == "HS":
        q_star, es_list = find_q_star(*hs_paths(b, c))
    elif name == "FHS":
        q_star, es_list = find_q_star(*fhs_paths(b, c))
    else:
        tb, tc, nu = t_paths(b, c)
        q_star, es_list = find_q_star(tb, tc)
    print(f"{name:6}  {es_list[0]:.2%}    {es_list[5]:.2%}       {es_list[10]:.2%}       "
          f"{es_list[20]:.2%}       {budget * es_list[0]:.2%}   {q_star:.2f}")
vol_b, today_b = ewma_vol(b)
print(f"base vol today (EWMA) = {today_b * np.sqrt(252):.1%}, "
      f"average of past 3 years = {b.std() * np.sqrt(252):.1%}")


# ---------------- 5. run every month ----------------
rows = []
bt_rows = []       # for the backtest in part 8
for i in range(window, len(dates) - future, step):
    b = base_ret[i - window:i]
    c = nvda_ret[i - window:i]

    hb, hc = hs_paths(b, c)
    fb, fc = fhs_paths(b, c)
    tb, tc, nu = t_paths(b, c)
    q_hs, _ = find_q_star(hb, hc)
    q_fhs, _ = find_q_star(fb, fc)
    q_t, _ = find_q_star(tb, tc)
    D = np.std([q_hs, q_fhs, q_t], ddof=1)          # disagreement

    # backtest: hold each model's q* for the next 10 days, what is the real loss?
    q_min = min(q_hs, q_fhs, q_t)                   # rule: take the most conservative q*
    for name, q, paths in [("HS", q_hs, (hb, hc)), ("FHS", q_fhs, (fb, fc)),
                           ("t-MC", q_t, (tb, tc)), ("min rule", q_min, None)]:
        if paths is None:
            var, es = np.nan, np.nan                # min rule is not one model
        else:
            var, es = get_var_es(paths[0], paths[1], q)
        port_next = (1 - q) * base_ret[i:i + horizon] + q * nvda_ret[i:i + horizon]
        real_loss = -(np.prod(1 + port_next) - 1)
        bt_rows.append([dates[i - 1], name, q, var, es, real_loss])

    # control: volatility of the probe portfolio in the past 20 days
    probe_past = (1 - probe_q) * b[-20:] + probe_q * c[-20:]
    vol20 = probe_past.std(ddof=1) * np.sqrt(252)

    # outcome: max drawdown of the probe portfolio in the next 20 days
    probe_next = (1 - probe_q) * base_ret[i:i + future] + probe_q * nvda_ret[i:i + future]
    wealth = np.concatenate([[1], np.cumprod(1 + probe_next)])   # start at 1 today
    mdd = (1 - wealth / np.maximum.accumulate(wealth)).max()

    rows.append([dates[i - 1], q_hs, q_fhs, q_t, D, vol20, mdd])

df = pd.DataFrame(rows, columns=["date", "q_HS", "q_FHS", "q_tMC", "D", "vol20", "future_mdd"])
df = df.set_index("date")
df.to_csv("results.csv")
print("\n", len(df), "monthly decisions from", df.index[0].date(), "to", df.index[-1].date())


# ---------------- 6. results ----------------
print("\n--- q* of each model ---")
lowest = df[["q_HS", "q_FHS", "q_tMC"]].idxmin(axis=1).value_counts(normalize=True)
for col in ["q_HS", "q_FHS", "q_tMC"]:
    print(f"{col:6} mean = {df[col].mean():.3f}, min = {df[col].min():.2f}, "
          f"max = {df[col].max():.2f}, lowest of 3 in {lowest.get(col, 0):.0%} of months")
print(f"D: mean = {df['D'].mean():.3f}, the 3 models give the same q* in "
      f"{(df['D'] == 0).mean():.0%} of months")

# sort the months into 5 groups by D
df["group"] = pd.qcut(df["D"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
table = df.groupby("group", observed=True)[["D", "vol20", "future_mdd"]].mean()
table["months"] = df.groupby("group", observed=True).size()
print("\n--- future 20-day max drawdown by D group (1 = low D, 5 = high D) ---")
print(table.round(4))

print("\n--- correlation ---")
print(df[["D", "vol20", "future_mdd"]].corr().round(3))

# regression, t values use Newey-West standard errors
print("\n--- regression: future_mdd on D, then add vol20 as control ---")
for cols in [["D"], ["vol20"], ["D", "vol20"]]:
    X = sm.add_constant(df[cols])
    fit = sm.OLS(df["future_mdd"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
    t_text = ", ".join(f"t({c}) = {fit.tvalues[c]:.2f}" for c in cols)
    print(f"{' + '.join(cols):10}  {t_text:32}  R2 = {fit.rsquared:.3f}")

print("\n--- 5 months with the highest D ---")
print(df.sort_values("D", ascending=False).head(5)[["q_HS", "q_FHS", "q_tMC", "D", "vol20", "future_mdd"]].round(3))

# check: is the result only from a few months?
print("\n--- check: is the result only from a few months? ---")
no2013 = df[df.index.year != 2013]
X = sm.add_constant(no2013[["D", "vol20"]])
fit = sm.OLS(no2013["future_mdd"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
print("without 2013: t(D) =", round(fit.tvalues["D"], 2), ", t(vol20) =", round(fit.tvalues["vol20"], 2))

df["D_rank"] = df["D"].rank() / len(df)      # rank of D, between 0 and 1
X = sm.add_constant(df[["D_rank", "vol20"]])
fit = sm.OLS(df["future_mdd"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
print("use D rank:   t(D_rank) =", round(fit.tvalues["D_rank"], 2), ", t(vol20) =", round(fit.tvalues["vol20"], 2))

# in the high D group, which model is far from the other two?
count = {"q_HS": 0, "q_FHS": 0, "q_tMC": 0}
for date, row in df[df["group"] == 5].iterrows():
    qs = row[["q_HS", "q_FHS", "q_tMC"]]
    far = (qs - qs.median()).abs().idxmax()
    count[far] += 1
print("model far from the other two in group 5:", count)


# ---------------- 7. backtest: which model's q* really kept the risk? ----------------
bt = pd.DataFrame(bt_rows, columns=["date", "model", "q", "VaR", "ES", "real_loss"])
bt.to_csv("backtest.csv", index=False)

print("\n--- backtest: hold each model's q* for the next 10 days ---")
print("VaR break = real 10-day loss is bigger than the model's 95% VaR (should be about 5%)")
summary = []
for name in ["HS", "FHS", "t-MC", "min rule"]:
    x = bt[bt["model"] == name]
    n = len(x)
    k = int(np.ceil(n * (1 - alpha)))
    worst = x["real_loss"].sort_values().values[-k:]      # the worst 5% of months
    if name == "min rule":
        breaks, break_rate, p_value, pred_es = "-", "-", "-", "-"
    else:
        hit = x["real_loss"] > x["VaR"]
        breaks = int(hit.sum())
        break_rate = f"{hit.mean():.1%}"
        p_value = round(binomtest(breaks, n, 1 - alpha).pvalue, 3)
        pred_es = f"{x['ES'].mean():.2%}"
    summary.append([name, round(x["q"].mean(), 3), f"{-x['real_loss'].mean():.2%}",
                    pred_es, f"{worst.mean():.2%}", f"{x['real_loss'].max():.2%}",
                    breaks, break_rate, p_value])
summary = pd.DataFrame(summary, columns=["model", "mean q", "mean 10d return", "mean model ES",
                                         "real ES (worst 5%)", "worst 10d loss",
                                         "VaR breaks", "break rate", "p-value"])
print(summary.to_string(index=False))
print("months:", n, ", expected VaR breaks if the model is right:", round(n * (1 - alpha), 1))


# ---------------- 8. plot ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(df.index, df["D"], color="black", linewidth=1)
ax[0].set_title("Model disagreement D over time")
ax[0].set_ylabel("D (SD of q* of the 3 models)")

ax[1].bar(table.index.astype(int), table["future_mdd"] * 100, color="tab:blue")
ax[1].set_title("Future 20-day max drawdown by D group")
ax[1].set_xlabel("D group (1 = low, 5 = high)")
ax[1].set_ylabel("mean future max drawdown (%)")

plt.tight_layout()
plt.savefig("figure.png", dpi=150)
print("\nsaved results.csv, backtest.csv and figure.png. Close the figure window to end the program.")
plt.show()
