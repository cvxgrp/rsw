import pandas as pd
import rsw
import matplotlib.pyplot as plt
import numpy as np


def get_dummies(df, column, prefix=None, columns=None):
    dummies = pd.get_dummies(df[column], prefix=prefix, columns=columns)
    dummies[df[column].isna()] = np.nan
    return dummies


def get_brfss():
    raw_df = pd.read_sas("data/LLCP2018.XPT ", format='xport')

    states = pd.read_csv("data/us_states.csv")
    states = states.set_index("post_code")

    columns = ["_STATE", "_AGE_G", "SEX1",
               "EDUCA", "INCOME2", "HTM4", "GENHLTH"]
    df = raw_df[columns]

    df._STATE.replace(dict(zip(states.fips, states.index)), inplace=True)
    df._STATE.replace({
        66: np.nan
    }, inplace=True)
    df._AGE_G.replace({
        1: "18_24",
        2: "25_34",
        3: "35_44",
        4: "45_54",
        5: "55_65",
        6: "65+"
    }, inplace=True)
    df.SEX1.replace({
        1: "M",
        2: "F",
        7: np.nan,
        9: np.nan
    }, inplace=True)
    df.EDUCA.replace({
        1: np.nan,
        2: "Elementary",
        3: np.nan,
        4: "High school",
        5: "Some college",
        6: "College",
        9: np.nan
    }, inplace=True)
    df.INCOME2.replace({
        1: "<10k",
        2: "<15k",
        3: "<20k",
        4: "<25k",
        5: "<35k",
        6: "<50k",
        7: "<75k",
        8: ">75k",
        77: np.nan,
        99: np.nan
    }, inplace=True)
    df.GENHLTH.replace({
        1: "Excellent",
        2: "Very good",
        3: "Good",
        4: "Fair",
        5: "Poor",
        7: np.nan,
        9: np.nan
    }, inplace=True)

    df.loc[:, "State Age"] = df._STATE + " " + df._AGE_G
    df.loc[:, "Education Income"] = df.EDUCA + " " + df.INCOME2

    df_processed = pd.concat([
        get_dummies(df, "State Age", "state_age"),
        get_dummies(df, "SEX1", "sex"),
        get_dummies(df, "Education Income", "education_income"),
        get_dummies(df, "GENHLTH", "health")
    ], axis=1)

    return raw_df, df, df_processed

# get data
print("Getting data.")
raw_df, df, df_processed = get_brfss()

# compute sample averages
print("Computing sample averages.")
fdes = df_processed.mean()
state_age_mean = fdes[filter(lambda x: "state_age" in x, df_processed.columns)]
sex_mean = fdes[filter(lambda x: "sex" in x, df_processed.columns)]
education_income_mean = fdes[
    filter(lambda x: "education_income" in x, df_processed.columns)]
health_mean = fdes[filter(lambda x: "health" in x, df_processed.columns)]

# construct skewed sample
print("Constructing skewed sample.")
df_for_sampling = df_processed.fillna(df_processed.mean())
np.random.seed(0)
n = 10000
while True:
    c = np.random.randn(df_processed.shape[1]) / 2
    pi = np.exp(df_for_sampling@c) / np.sum(np.exp(df_for_sampling@c))
    idx = np.random.choice(df_for_sampling.shape[
                           0], p=pi, size=n, replace=False)
    df_small = df_processed.loc[idx]
    if (df_small.mean() > 0).all():
        break

# Maximum entropy weighting
print("\n\n Maximum entropy weighting")
losses = [
    rsw.EqualityLoss(np.array(state_age_mean).flatten()),
    rsw.EqualityLoss(np.array(sex_mean).flatten()),
    rsw.EqualityLoss(np.array(education_income_mean).flatten()),
    rsw.EqualityLoss(np.array(health_mean).flatten())
]
regularizer = rsw.EntropyRegularizer(limit=100)
w, out, sol = rsw.rsw(df_small, None, losses, regularizer,
                      1, verbose=True, rho=75, eps_abs=1e-6, eps_rel=1e-6)

w = np.clip(w, 1 / (100 * n), 100 / n)
w /= np.sum(w)
w_maxent = w
n = w.size
t = df[:n]
t["weight"] = w
print(t.loc[t.weight.argmax()])
# print(t[t["State Age"] == "NJ 45_54"])
print(-np.sum(w * np.log(w)), -np.sum(np.ones(n) / n * np.log(np.ones(n) / n)))
print(w.min(), w.max())
plt.hist(w, bins=100, color='black')
plt.xlabel("$w_i$")
plt.ylabel("count")
plt.savefig("figs/hist_w.pdf")
plt.close()

x = np.array(raw_df['HTIN4'].iloc[idx])
x = x[~np.isnan(x)]
hist_unweighted, vals = np.histogram(x, bins=1000)
cdf_unweighted = np.cumsum(hist_unweighted) / np.sum(hist_unweighted)
plt.plot(vals[1:], cdf_unweighted, label='unweighted', c='grey', linewidth=1)

x = np.array(raw_df['HTIN4'].iloc[idx])
w_adjusted = w * 1 / (1 - w[np.isnan(x)].sum())
w_adjusted[np.isnan(x)] = 0
hist_weighted, _ = np.histogram(x, bins=vals, weights=w_adjusted)
cdf_weighted = np.cumsum(hist_weighted) / np.sum(hist_weighted)
plt.plot(vals[1:], cdf_weighted, label='weighted', c='black', linewidth=1)

x = np.array(raw_df['HTIN4'])
x = x[~np.isnan(x)]
hist_true, _ = np.histogram(x, bins=vals)
cdf_true = np.cumsum(hist_true) / np.sum(hist_true)
plt.plot(vals[1:], cdf_true, '--', label='true', c='black', linewidth=1)

plt.xlim(55, 80)
plt.xlabel("Height (inches)")
plt.ylabel("CDF")
plt.legend()
plt.savefig("figs/height.pdf")
plt.close()

print("%3.3f, %3.3f" % (np.abs(cdf_weighted - cdf_true).max(),
                        np.abs(cdf_unweighted - cdf_true).max()))

x = np.array(raw_df['WTKG3'].iloc[idx]) / 100
x = x[~np.isnan(x)]
hist_unweighted, vals = np.histogram(x, bins=1000)
cdf_unweighted = np.cumsum(hist_unweighted) / np.sum(hist_unweighted)
plt.plot(vals[1:], cdf_unweighted, label='unweighted', c='grey', linewidth=1)

x = np.array(raw_df['WTKG3'].iloc[idx]) / 100
w_adjusted = w * 1 / (1 - w[np.isnan(x)].sum())
w_adjusted[np.isnan(x)] = 0
hist_weighted, _ = np.histogram(x, bins=vals, weights=w_adjusted)
cdf_weighted = np.cumsum(hist_weighted) / np.sum(hist_weighted)
plt.plot(vals[1:], cdf_weighted, label='weighted', c='black', linewidth=1)

x = np.array(raw_df['WTKG3']) / 100
x = x[~np.isnan(x)]
hist_true, _ = np.histogram(x, bins=vals)
cdf_true = np.cumsum(hist_true) / np.sum(hist_true)
plt.plot(vals[1:], cdf_true, '--', label='true', c='black', linewidth=1)

plt.xlim(30, 140)
plt.xlabel("Weight (kg)")
plt.ylabel("CDF")
plt.legend()
plt.savefig("figs/weight.pdf")
plt.close()

print("%3.3f, %3.3f" % (np.abs(cdf_weighted - cdf_true).max(),
                        np.abs(cdf_unweighted - cdf_true).max()))

# Representative selection
print("\n\nRepresentative selection")
losses = [
    rsw.KLLoss(np.array(state_age_mean).flatten(), scale=1),
    rsw.KLLoss(np.array(sex_mean).flatten(), scale=1),
    rsw.KLLoss(np.array(education_income_mean).flatten(), scale=1),
    rsw.KLLoss(np.array(health_mean).flatten(), scale=1)
]
regularizer = rsw.BooleanRegularizer(k=500)

w, out, sol = rsw.rsw(df_small, None, losses, regularizer, 1,
                      verbose=True, rho=100, eps_abs=1e-5, eps_rel=1e-5, maxiter=1000)

ct_cum = 0
objective_ours = 0.
f = np.concatenate(out)
for l in losses:
    objective_ours += l.evaluate(f[ct_cum:ct_cum + l.m])
    ct_cum += l.m

np.random.seed(0)
lz = []
idxs = []
for _ in range(200):
    ct_cum = 0
    objective = 0.
    idxtemp = np.random.choice(
        df_small.index, size=regularizer.k, p=w_maxent, replace=False)
    idxs.append(idxtemp)
    f = np.array(df_small.loc[idxtemp].mean())
    for l in losses:
        objective += l.evaluate(f[ct_cum:ct_cum + l.m])
        ct_cum += l.m
    lz.append(objective)

print(1 - objective_ours / np.min(lz))

plt.xlabel('$\\ell(f, f^\\mathrm{des})$')
plt.hist(lz, bins=50, color='grey', label='random')
plt.axvline(objective_ours, color='black', label='solution')
plt.legend()
plt.ylabel("count")
plt.savefig("figs/boolean_hist.pdf")
plt.close()

kss = []
for idxtemp in idxs:
    w_unweighted = np.zeros(df_small.shape[0])
    where_1 = df_small.index.isin(idxtemp)
    w_unweighted[where_1] = 1 / regularizer.k

    x = np.array(raw_df['HTIN4'].iloc[idx])
    w_adjusted = w_unweighted * 1 / (1 - w_unweighted[np.isnan(x)].sum())
    w_adjusted[np.isnan(x)] = 0
    x[np.isnan(x)] = 0
    hist_unweighted, vals = np.histogram(
        x, bins=np.arange(0, 200, 1), weights=w_adjusted)
    cdf_unweighted = np.cumsum(hist_unweighted) / np.sum(hist_unweighted)

    x = np.array(raw_df['HTIN4'].iloc[idx])
    w_adjusted = w * 1 / (1 - w[np.isnan(x)].sum())
    w_adjusted[np.isnan(x)] = 0
    hist_weighted, _ = np.histogram(x, bins=vals, weights=w_adjusted)
    cdf_weighted = np.cumsum(hist_weighted) / np.sum(hist_weighted)

    x = np.array(raw_df['HTIN4'])
    x = x[~np.isnan(x)]
    hist_true, _ = np.histogram(x, bins=vals)
    cdf_true = np.cumsum(hist_true) / np.sum(hist_true)

    kss.append(np.abs(cdf_unweighted - cdf_true).max())

ks = np.abs(cdf_weighted - cdf_true).max()

print(((ks - np.array(kss)) <= 0).mean())
plt.hist(kss, bins=50, color='gray', label='random')
plt.axvline(ks, color='black', label='selection')
plt.legend()
plt.xlabel("K-S test statistic")
plt.ylabel("count")
plt.savefig("figs/ks_boolean_height.pdf")
plt.close()

kss = []
for idxtemp in idxs:
    w_unweighted = np.zeros(df_small.shape[0])
    where_1 = df_small.index.isin(idxtemp)
    w_unweighted[where_1] = 1 / regularizer.k

    x = np.array(raw_df['WTKG3'].iloc[idx] / 100)
    w_adjusted = w_unweighted * 1 / (1 - w_unweighted[np.isnan(x)].sum())
    w_adjusted[np.isnan(x)] = 0
    x[np.isnan(x)] = 0
    hist_unweighted, vals = np.histogram(
        x, bins=np.arange(0, 200, 1), weights=w_adjusted)
    cdf_unweighted = np.cumsum(hist_unweighted) / np.sum(hist_unweighted)

    x = np.array(raw_df['WTKG3'].iloc[idx] / 100)
    w_adjusted = w * 1 / (1 - w[np.isnan(x)].sum())
    w_adjusted[np.isnan(x)] = 0
    hist_weighted, _ = np.histogram(x, bins=vals, weights=w_adjusted)
    cdf_weighted = np.cumsum(hist_weighted) / np.sum(hist_weighted)

    x = np.array(raw_df['WTKG3'] / 100)
    x = x[~np.isnan(x)]
    hist_true, _ = np.histogram(x, bins=vals)
    cdf_true = np.cumsum(hist_true) / np.sum(hist_true)

    kss.append(np.abs(cdf_unweighted - cdf_true).max())

ks = np.abs(cdf_weighted - cdf_true).max()

print(((ks - np.array(kss)) <= 0).mean())
plt.hist(kss, bins=50, color='gray', label='random')
plt.axvline(ks, color='black', label='selection')
plt.legend()
plt.xlabel("K-S test statistic")
plt.ylabel("count")
plt.savefig("figs/ks_boolean_weight.pdf")
plt.close()
