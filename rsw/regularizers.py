import cvxpy as cp
import numpy as np
from scipy.special import lambertw


class ZeroRegularizer():

    def __init__(self):
        pass

    def prox(self, w, lam):
        return w


class EntropyRegularizer():

    def __init__(self, limit=None):
        if limit is not None and limit <= 1:
            raise ArgumentError("limit is %.3f. It must be > 1." % limit)
        self.limit = limit

    def prox(self, w, lam):
        what = lam * np.real(lambertw(np.exp(w / lam - 1) / lam, tol=1e-12))
        if self.limit is not None:
            what = np.clip(what, 1 / (self.limit * w.size),
                           self.limit / w.size)
        return what


class KLRegularizer():

    def __init__(self, prior, w_min=0, w_max=float("inf")):
        self.prior = prior
        self.entropy_reg = EntropyRegularizer(w_min, w_max)

    def prox(self, w, lam):
        return self.entropy_reg.prox(w + lam * np.log(self.prior), lam)


class CardinalityRegularizer():

    def __init__(self, k):
        raise NotImplementedError
        self.k = k

    def prox(self, w, lam):
        out = np.copy(w)
        idx = np.argsort(w)[:-self.k]
        out[idx] = 0.
        return out


class BooleanRegularizer():

    def __init__(self, k):
        self.k = k

    def prox(self, w, lam):
        idx_sort = np.argsort(w)
        new_arr = np.zeros(len(w))
        new_arr[idx_sort[-self.k:]] = 1. / self.k
        return new_arr

if __name__ == "__main__":
    w = np.random.randn(10)
    prior = np.random.uniform(10)
    prior /= 10
    lam = .5
    zero_reg = ZeroRegularizer()
    np.testing.assert_allclose(zero_reg.prox(w, .5), w)

    entropy_reg = EntropyRegularizer()
    what = cp.Variable(10)
    cp.Problem(cp.Minimize(-cp.sum(cp.entr(what)) + 1 /
                           (2 * lam) * cp.sum_squares(what - w))).solve()
    np.testing.assert_allclose(what.value, entropy_reg.prox(w, lam), atol=1e-4)

    kl_reg = KLRegularizer(prior)
    what = cp.Variable(10)
    cp.Problem(cp.Minimize(-cp.sum(cp.entr(what)) - cp.sum(cp.multiply(what, np.log(prior))) + 1 /
                           (2 * lam) * cp.sum_squares(what - w))).solve()
    np.testing.assert_allclose(what.value, kl_reg.prox(w, lam), atol=1e-4)
