"""Mechanism-class LP upper/lower bound -- DIAGNOSTIC ONLY (no finite-sample guarantee claimed).

Fixes relative to envelope.py (Codex round-3 item 4):
  * dictionary CDFs computed exactly (FFT, no Monte-Carlo error) for constant-(eps, P) documents;
  * the least-favourable member P* = (1-delta, delta), eps = eps_min, is in the dictionary;
  * both 'rest spread over 9 tokens' and 'rest concentrated in one token' NTP shapes are included.
What remains unproved: a document whose q_t varies within the document is NOT in the convex hull of constant-q
documents, so the LP region is an inner approximation of the mechanism class.  We therefore report empirical
coverage of the LP interval on (i) the within-document-varying in-class DGP and (ii) the exact least-favourable DGP.
Run: python3 lp_diag.py --out ../figures [--reps 100]
"""
import argparse, json, os, time
import numpy as np
from scipy import stats
from envelope import F1_P, x_density_on_grid, convolve_power, lp_mech_bounds
from certified import certified_envelope, band_bounds
from dgp import draw_docs, draw_docs_const

def doc_cdf(FY, n, ts, ds=0.005, smax_tok=40.0):
    """P(p <= t) for a document of n iid pivots with CDF FY (FFT, midpoint grid; approx, for diagnostics)."""
    s, pmf = x_density_on_grid(FY, ds, smax_tok)
    mean = (s * pmf).sum(); sd = np.sqrt(((s - mean) ** 2 * pmf).sum())
    L = int(2 ** np.ceil(np.log2((n * mean + 12 * sd * np.sqrt(n) + 50) / ds)))
    g = convolve_power(pmf, n, L); cdf = np.cumsum(g); grid = np.arange(L) * ds + n * ds / 2.0
    return np.array([1 - cdf[min(np.searchsorted(grid, stats.gamma.ppf(1 - t, a=n)), L - 1)] for t in ts])

def build_dictionary(n, ts, eps_min, delta, eps_grid=(0.3, 0.4, 0.5, 0.6, 0.8, 1.0), q_grid=(0.3, 0.5, 0.7, 0.8)):
    members = []
    for q in q_grid:
        members.append(("spread9", np.r_[q, np.full(9, (1 - q) / 9)]))
        if 1 - q <= q: members.append(("conc", np.r_[q, 1 - q]))
    members.append(("Pstar", np.r_[1 - delta, delta]))
    rows = []; labels = []
    for eps in eps_grid:
        for name, P in members:
            FY = lambda x, eps=eps, P=P: (1 - eps) * x + eps * F1_P(x, P)
            rows.append(doc_cdf(FY, n, ts)); labels.append((eps, name, float(P[0])))
    return np.array(rows), labels

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../figures"); ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args(); rng = np.random.default_rng(13)
    ts = np.linspace(0.01, 0.9, 90)
    n, N, pi, em, d = 500, 20000, 0.2, 0.3, 0.2
    t0 = time.time(); D, labels = build_dictionary(n, ts, em, d); print("dictionary", D.shape, round(time.time() - t0, 1), "s", flush=True)
    hl = certified_envelope(n, em, d, ts); band = np.sqrt(np.log(2 / 0.05) / (2 * N))
    res = {"n": n, "N": N, "pi": pi, "assumed": [em, d], "dict_size": len(labels), "band": band, "cases": []}
    for case, drawer in (("in_class_varying_q", lambda: draw_docs(N, n, pi, rng)),
                         ("least_favourable_Pstar", lambda: draw_docs_const(N, n, pi, rng, q_const=1 - d, eps_const=em, nrest=1))):
        covL = covR = 0; los = []; his = []; rl = []; ru = []
        for r in range(a.reps):
            p, _ = drawer(); F = np.array([(p <= t).mean() for t in ts])
            lo, hi = lp_mech_bounds(F, ts, D, band); lcb, ucb = band_bounds(p, hl, ts)
            covL += (lo <= pi <= hi); covR += (lcb <= pi <= ucb); los.append(lo); his.append(hi); rl.append(lcb); ru.append(ucb)
        row = {"case": case, "reps": a.reps, "LP_coverage": covL / a.reps, "LP_mean_interval": [round(float(np.nanmean(los)), 4), round(float(np.nanmean(his)), 4)],
               "LP_min_hi": round(float(np.nanmin(his)), 4), "relaxed_coverage": covR / a.reps, "relaxed_mean_interval": [round(float(np.mean(rl)), 4), round(float(np.mean(ru)), 4)]}
        res["cases"].append(row); print(row, flush=True)
    with open(os.path.join(a.out, "lp_diag_results.json"), "w") as f:
        json.dump(res, f, indent=1)

if __name__ == "__main__":
    main()
