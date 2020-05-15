import cvxpy as cp
import numpy as np
from scipy.special import lambertw, kl_div
from numbers import Number


class EqualityLoss():

    def __init__(self, fdes):
        if isinstance(fdes, Number):
            fdes = np.array([fdes])
        self.fdes = fdes
        self.m = fdes.size

    def prox(self, f, lam):
        return self.fdes


class InequalityLoss():

    def __init__(self, fdes, lower, upper):
        if isinstance(fdes, Number):
            fdes = np.array([fdes])
        self.fdes = fdes
        self.m = fdes.size
        self.lower = lower
        self.upper = upper
        assert (self.lower <= self.upper).all()

    def prox(self, f, lam):
        return np.clip(f, self.fdes + self.lower, self.fdes + self.upper)


class LeastSquaresLoss():

    def __init__(self, fdes, diag_weight=None):
        if isinstance(fdes, Number):
            fdes = np.array([fdes])
        self.fdes = fdes
        self.m = fdes.size
        if diag_weight is None:
            diag_weight = 1.
        self.diag_weight = diag_weight

    def prox(self, f, lam):
        return (self.diag_weight**2 * self.fdes + f / lam) / (self.diag_weight**2 + 1 / lam)

    def evaluate(self, f):
        return np.sum(np.square(self.diag_weight * (f - self.fdes)))


def _entropy_prox(f, lam):
    return lam * np.real(lambertw(np.exp(f / lam - 1) / lam, tol=1e-10))


class KLLoss():

    def __init__(self, fdes, scale=1):
        if isinstance(fdes, Number):
            fdes = np.array([fdes])
        self.fdes = fdes
        self.m = fdes.size
        self.scale = scale

    def prox(self, f, lam):
        return _entropy_prox(f + lam * self.scale * np.log(self.fdes), lam * self.scale)

    def evaluate(self, f):
        return self.scale * np.sum(kl_div(f, self.fdes))

if __name__ == "__main__":
    m = 10
    f = np.random.randn(m)
    fdes = np.random.randn(m)
    lam = 1

    equality = EqualityLoss(fdes)
    fhat = cp.Variable(m)
    cp.Problem(cp.Minimize(1 / lam * cp.sum_squares(fhat - f)),
               [fhat == fdes]).solve()
    np.testing.assert_allclose(fhat.value, equality.prox(f, lam))

    lower = np.array([-.3])
    upper = np.array([.3])
    inequality = InequalityLoss(fdes, lower, upper)
    fhat = cp.Variable(m)
    cp.Problem(cp.Minimize(1 / lam * cp.sum_squares(fhat - f)),
               [lower <= fhat - fdes, fhat - fdes <= upper]).solve()
    np.testing.assert_allclose(fhat.value, inequality.prox(f, lam))

    d = np.random.uniform(0, 1, size=m)
    lstsq = LeastSquaresLoss(fdes, d)
    fhat = cp.Variable(m)
    cp.Problem(cp.Minimize(1 / 2 * cp.sum_squares(cp.multiply(d, fhat - fdes)) +
                           1 / (2 * lam) * cp.sum_squares(fhat - f))).solve()
    np.testing.assert_allclose(fhat.value, lstsq.prox(f, lam))

    f = np.random.uniform(0, 1, size=m)
    f /= f.sum()
    fdes = np.random.uniform(0, 1, size=m)
    fdes /= fdes.sum()

    fhat = cp.Variable(m)
    cp.Problem(cp.Minimize(cp.sum(-cp.entr(fhat)) +
                           1 / (2 * lam) * cp.sum_squares(fhat - f))).solve()
    np.testing.assert_allclose(
        fhat.value, _entropy_prox(f, lam), atol=1e-5)

    kl = KLLoss(fdes, scale=.5)
    fhat = cp.Variable(m, nonneg=True)
    cp.Problem(cp.Minimize(.5 * (cp.sum(-cp.entr(fhat) - cp.multiply(fhat, np.log(fdes)))) +
                           1 / (2 * lam) * cp.sum_squares(fhat - f))).solve()
    np.testing.assert_allclose(fhat.value, kl.prox(f, lam), atol=1e-5)
