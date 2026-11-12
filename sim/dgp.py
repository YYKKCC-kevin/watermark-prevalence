"""Shared synthetic DGPs (document-level p-values) for the C3 simulations.

Gumbel-max pivots, sum statistic T_n = sum -log(1-Y), p = Gamma_n.sf(T_n).
  draw_docs(...)        fixed in-class DGP used in certified.py: q_t ~ U[q_lo,q_hi] per token, eps_i ~ U[eps_lo,1]
  draw_docs_const(...)  sharp out-of-class DGP: q_t == q_const, eps_i == eps_const
"""
import numpy as np
from scipy import stats

def _wm_pvalues(k1, n, q, eps, rng, nrest=9):
    """p-values of k1 watermarked docs; q: (k1,n) top probabilities, eps: (k1,) retention."""
    rest = (1 - q) / nrest
    pick = rng.random((k1, n)) < q
    pw = np.where(pick, q, rest)
    Yw = rng.random((k1, n)) ** pw                  # Y | w ~ U^{P(w)}
    ret = rng.random((k1, n)) < eps[:, None]        # Model *: Bernoulli retention switch
    Y = np.where(ret, Yw, rng.random((k1, n)))
    return stats.gamma.sf((-np.log1p(-Y)).sum(1), a=n)

def draw_docs(N, n, pi, rng, eps_lo=0.3, q_lo=0.3, q_hi=0.8, Z=None, nrest=9):
    """Return (p, Z). If Z is given it is used as the latent labels (allows arbitrary/adaptive Z)."""
    if Z is None:
        Z = rng.random(N) < pi
    Z = np.asarray(Z, bool)
    p = np.empty(N)
    k0 = int((~Z).sum()); k1 = int(Z.sum())
    if k0:
        p[~Z] = stats.gamma.sf(rng.gamma(n, 1.0, k0), a=n)
    if k1:
        eps = rng.uniform(eps_lo, 1.0, k1)
        q = rng.uniform(q_lo, q_hi, (k1, n))
        p[Z] = _wm_pvalues(k1, n, q, eps, rng, nrest=nrest)
    return p, Z

def draw_docs_const(N, n, pi, rng, q_const=0.9, eps_const=0.3, Z=None, nrest=9):
    if Z is None:
        Z = rng.random(N) < pi
    Z = np.asarray(Z, bool)
    p = np.empty(N)
    k0 = int((~Z).sum()); k1 = int(Z.sum())
    if k0:
        p[~Z] = stats.gamma.sf(rng.gamma(n, 1.0, k0), a=n)
    if k1:
        q = np.full((k1, n), q_const); eps = np.full(k1, eps_const)
        p[Z] = _wm_pvalues(k1, n, q, eps, rng, nrest=nrest)
    return p, Z
