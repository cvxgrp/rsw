import pandas as pd
import numpy as np
from numpy import linalg
import rsw

np.random.seed(5)
n = 100
age = np.random.randint(20, 30, size=n) * 1.
sex = np.random.choice([0., 1.], p=[.4, .6], size=n)
height = np.random.normal(5, 1, size=n)

df = pd.DataFrame({
    "age": age,
    "sex": sex,
    "height": height
})

# Real
print("\n\nExample 1, Max Entropy Weights")
funs = [
    lambda x: x.age,
    lambda x: x.sex == 0 if not np.isnan(x.sex) else np.nan,
    lambda x: x.height
]
losses = [rsw.EqualityLoss(25), rsw.EqualityLoss(.5),
          rsw.EqualityLoss(5.3)]
regularizer = rsw.EntropyRegularizer()
w, out, sol = rsw.rsw(df, funs, losses, regularizer, 1., verbose=True)
df["weight"] = w
print(df.head())
print(out)

# Real
print("\n\nExample 2, Boolean Weights")
funs = [
    lambda x: x.age,
    lambda x: x.sex == 0 if not np.isnan(x.sex) else np.nan,
    lambda x: x.height
]
losses = [rsw.LeastSquaresLoss(25), rsw.LeastSquaresLoss(.5),
          rsw.LeastSquaresLoss(5.3)]
regularizer = rsw.BooleanRegularizer(5)
w, out, sol = rsw.rsw(df, funs, losses, regularizer, 1., verbose=True)
df["weight"] = w
print(df[df.weight > .1])
print(out)

# nans
print("\n\nExample 3, Missing values")
for i, j in zip(np.random.randint(50, size=25), np.random.randint(3, size=25)):
    df.iat[i, j] *= np.nan
# Real
funs = [
    lambda x: x.age,
    lambda x: x.sex == 0 if not np.isnan(x.sex) else np.nan,
    lambda x: x.height
]
losses = [rsw.EqualityLoss(25), rsw.EqualityLoss(.5),
          rsw.EqualityLoss(5.3)]
regularizer = rsw.EntropyRegularizer()
w, out, sol = rsw.rsw(df, funs, losses, regularizer, 1.)
df["weight"] = w
print(df.head())
print(out)
