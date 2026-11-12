"""Certified (one-sided) power envelope and coverage experiment.

h_lower(t) <= h(t) := P(T̄_n >= Gamma_n^{-1}(1-t)) with ALL approximation errors pushed downward:
  * X = -log(1-Ȳ) replaced by X_tr = min(floor(X/ds)*ds, S_max) <= X  (downward quantisation + truncation)
  * exact lattice pmf of X_tr from the closed-form CDF of X
  * n-fold convolution on a lattice long enough to hold the FULL support (no wrap-around)
  * threshold c rounded UP to the lattice; a numerical tolerance subtracted for FFT round-off
Then P(T̄_n >= c) >= P(sum X_tr >= ceil(c/ds)*ds) >= h_lower(t).

Coverage experiment on a FIXED DGP (independent of the assumed (eps_min, delta)):
  q_t ~ U[0.3, 0.8] per token, eps_i ~ U[0.3, 1], so the DGP lies in A(eps_min, delta) for eps_min<=0.3, delta<=0.2.

Run: python3 -I certified.py --out ../figures [--reps 100]
"""
import argparse, json, os, time
import numpy as np
from scipy import stats
from envelope import F_bar, sample_docs, power_envelope

FFT_TOL = 1e-8

def certified_envelope(n, eps_min, delta, ts, ds=0.01, smax=30.0, sigma=0.0):
    """sigma: share of scored positions that are substitutions at intact contexts (worst-case law F_sub(.; 1-delta),
    see substitution.py).  sigma=0 reproduces the original envelope."""
    assert 0 <= sigma and eps_min + sigma <= 1
    K = int(round(smax / ds))
    s = np.arange(K + 1) * ds
    xx = 1 - np.exp(-s)
    FX = F_bar(xx, eps_min, delta)                         # CDF of X at lattice points
    if sigma > 0:
        from substitution import F_sub
        FX = FX + sigma * (F_sub(xx, 1 - delta) - xx)      # mixture weights (1-eps-sigma, eps, sigma)
    pmf = np.empty(K + 1)
    pmf[:K] = np.diff(FX)                                  # mass of [k ds, (k+1) ds) -> k ds
    pmf[K] = 1.0 - FX[K]                                   # tail mass -> S_max
    assert pmf.min() >= -1e-15 and abs(pmf.sum() - 1) < 1e-12
    L = 1 << int(np.ceil(np.log2(n * K + 1)))              # full support fits: no wrap-around
    f = np.fft.rfft(pmf, L)
    g = np.fft.irfft(f ** n, L)[: n * K + 1]
    g = np.clip(g, 0, None)
    tail = np.cumsum(g[::-1])[::-1]                        # tail[k] = P(sum >= k ds) (+ round-off)
    out = []
    for t in ts:
        c = stats.gamma.ppf(1 - t, a=n) * (1 + 1e-9)
        k = int(np.ceil(c / ds))
        v = tail[k] if k <= n * K else 0.0
        out.append(max(v - FFT_TOL, 0.0))
    return np.array(out)

def band_bounds(p, hl, ts, alpha=0.05):
    N = len(p); b = np.sqrt(np.log(2 / alpha) / (2 * N))
    F = np.array([(p <= t).mean() for t in ts])
    lcb = max(np.max((F - b - ts) / (1 - ts)), 0.0)
    m = hl - ts > 0
    ucb = min(np.min(((F + b - ts) / (hl - ts))[m]), 1.0) if m.any() else 1.0
    return lcb, ucb

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../figures"); ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(7)
    ts = np.concatenate([np.array([0.001, 0.005]), np.linspace(0.01, 0.99, 99)])
    res = {}

    # 1. certified vs FFT-approx envelope
    t0 = time.time(); comp = []
    for (n, em, d) in [(200, 0.3, 0.1), (500, 0.3, 0.2), (1000, 0.2, 0.05)]:
        hl = certified_envelope(n, em, d, ts); ha, _ = power_envelope(n, em, d, ts)
        i = [np.argmin(abs(ts - v)) for v in (0.01, 0.05, 0.1, 0.5)]
        comp.append({"n": n, "eps_min": em, "delta": d, "t": ts[i].round(3).tolist(), "h_lower": hl[i].round(4).tolist(), "h_fft_approx": ha[i].round(4).tolist(), "max_gap": float((ha - hl).max().round(4)), "lower_le_approx_everywhere": bool((hl <= ha + 1e-6).all())})
        print(comp[-1], flush=True)
    res["certified_vs_approx"] = comp; print("envelopes", round(time.time() - t0, 1), "s")

    # 2. coverage on a fixed DGP
    dgp = dict(eps_lo=0.3, q_lo=0.3, q_hi=0.8)
    def draw(N, n, pi):
        # same as envelope.sample_docs but with fixed q range
        Z = rng.random(N) < pi; p = np.empty(N)
        k0 = (~Z).sum()
        if k0: p[~Z] = stats.gamma.sf(rng.gamma(n, 1.0, k0), a=n)
        k1 = Z.sum()
        if k1:
            eps = rng.uniform(dgp["eps_lo"], 1.0, k1)
            q = rng.uniform(dgp["q_lo"], dgp["q_hi"], (k1, n)); rest = (1 - q) / 9.0
            pick = rng.random((k1, n)) < q; pw = np.where(pick, q, rest)
            Yw = rng.random((k1, n)) ** pw; ret = rng.random((k1, n)) < eps[:, None]
            Y = np.where(ret, Yw, rng.random((k1, n)))
            p[Z] = stats.gamma.sf((-np.log1p(-Y)).sum(1), a=n)
        return p
    settings = [  # (assumed eps_min, assumed delta, in_class?)
        (0.3, 0.2, True), (0.3, 0.1, True), (0.2, 0.2, True), (0.3, 0.3, False)]
    rows = []
    for n in (200, 500):
        hls = {(em, d): certified_envelope(n, em, d, ts) for (em, d, _) in settings}
        for pi in (0.05, 0.2):
            for N in (2000, 20000):
                cov = {k: 0 for k in hls}; lc = {k: [] for k in hls}; uc = {k: [] for k in hls}
                for r in range(a.reps):
                    p = draw(N, n, pi)
                    for key, hl in hls.items():
                        lcb, ucb = band_bounds(p, hl, ts)
                        cov[key] += (lcb <= pi <= ucb); lc[key].append(lcb); uc[key].append(ucb)
                for (em, d, ok) in settings:
                    key = (em, d)
                    rows.append({"n": n, "pi": pi, "N": N, "assumed_eps_min": em, "assumed_delta": d, "dgp_in_class": ok, "coverage": cov[key] / a.reps, "mean_lcb": round(float(np.mean(lc[key])), 3), "mean_ucb": round(float(np.mean(uc[key])), 3), "reps": a.reps})
                    print(rows[-1], flush=True)
    res["coverage_fixed_dgp"] = rows; res["dgp"] = dgp
    with open(os.path.join(a.out, "certified_coverage_results.json"), "w") as f:
        json.dump(res, f, indent=1)

if __name__ == "__main__":
    main()
