"""Go/no-go numerics for Gumbel-max watermark prevalence identification.

Computes
  F*(x; delta)        least-favourable per-pivot alternative CDF (two-point NTP)
  Fbar(x)             least-favourable retained-mixture CDF
  h(t; n, eps_min, delta)  power envelope of the sum statistic T_n = sum -log(1-Y)
                      via FFT convolution of the per-token law (exact up to grid)
  pi_L, pi_U          relaxed-class identification bounds for a synthetic mixture
  LP mechanism-class upper bound with a finite dictionary.

Run:  python3 -I envelope.py --out ../figures
"""
import argparse, json, os, sys, time
import numpy as np
from scipy import stats, optimize

# ---------- least favourable laws ----------
def F_star(x, delta):
    x = np.asarray(x, float)
    return (1 - delta) * x ** (1 / (1 - delta)) + delta * x ** (1 / delta)

def F_bar(x, eps_min, delta):
    x = np.asarray(x, float)
    return (1 - eps_min) * x + eps_min * F_star(x, delta)

def F1_P(x, P):
    """Alternative pivot CDF for an arbitrary NTP vector P (sum_w P(w) x^{1/P(w)})."""
    x = np.asarray(x, float)[:, None]
    P = np.asarray(P, float)[None, :]
    P = P[:, P[0] > 0]
    return (P * x ** (1 / P)).sum(1)

# ---------- distribution of X = -log(1-Y) under a CDF F_Y, on a grid ----------
def x_density_on_grid(FY, ds, smax):
    s = np.arange(0, smax, ds)
    cdf = FY(1 - np.exp(-s))
    cdf = np.append(cdf, 1.0)
    pmf = np.diff(cdf)                      # mass in [s_k, s_{k+1})
    return s, pmf

def convolve_power(pmf, n, total_len):
    """n-fold convolution of a pmf on a uniform grid via FFT."""
    L = total_len
    f = np.fft.rfft(pmf, L)
    g = np.fft.irfft(f ** n, L)
    g = np.clip(g, 0, None)
    g /= g.sum()
    return g

def power_envelope(n, eps_min, delta, ts, ds=0.005, smax_tok=40.0):
    """h(t; n, eps_min, delta) for each t in ts: P(T̄_n >= Gamma_n^{-1}(1-t))."""
    s, pmf = x_density_on_grid(lambda x: F_bar(x, eps_min, delta), ds, smax_tok)
    # support of sum: up to n*smax_tok but mass concentrates near n*mean; use padded length
    mean = (s * pmf).sum(); sd = np.sqrt(((s - mean) ** 2 * pmf).sum())
    Smax = n * mean + 12 * sd * np.sqrt(n) + 50
    L = int(2 ** np.ceil(np.log2(Smax / ds)))
    g = convolve_power(pmf, n, L)
    cdf = np.cumsum(g)
    grid = np.arange(L) * ds + n * ds / 2.0   # midpoint correction
    out = []
    for t in ts:
        c = stats.gamma.ppf(1 - t, a=n)
        k = np.searchsorted(grid, c)
        out.append(1 - cdf[min(k, L - 1)])
    return np.array(out), mean - 1.0

def power_envelope_mc(n, eps_min, delta, ts, N=200000, rng=None):
    rng = rng or np.random.default_rng(0)
    # sample Ybar: with prob 1-eps_min uniform, else from F*: w.p. 1-delta U^{1-delta}, else U^{delta}
    U = rng.random((N, n)); B = rng.random((N, n)) < eps_min; C = rng.random((N, n)) < (1 - delta)
    Y = np.where(B, np.where(C, U ** (1 - delta), U ** delta), U)
    T = (-np.log1p(-Y)).sum(1)
    p = stats.gamma.sf(T, a=n)
    return np.array([(p <= t).mean() for t in ts])

# ---------- synthetic heterogeneous DGP (not least favourable) ----------
def sample_docs(N, n, pi, eps_dist, entropy_mode, rng, qmax=0.9):
    """Return p-values of N docs of length n; watermarked w.p. pi with heterogeneous eps and random NTPs."""
    Z = rng.random(N) < pi
    p = np.empty(N)
    # null docs
    k0 = (~Z).sum()
    if k0:
        T0 = rng.gamma(n, 1.0, size=k0); p[~Z] = stats.gamma.sf(T0, a=n)
    k1 = Z.sum()
    if k1:
        eps = eps_dist(k1, rng)
        # per-token top prob q_t drawn to mimic varying entropy; NTP = (q, rest spread over 9 tokens)
        if entropy_mode == "high":
            q = rng.uniform(0.2, 0.7, size=(k1, n))
        elif entropy_mode == "low":
            q = rng.uniform(0.6, qmax, size=(k1, n))
        else:
            q = rng.uniform(0.3, qmax, size=(k1, n))
        # sample Y from F_{1,P}: choose w by P, then Y = U_w | selected  ~ CDF x^{1/P(w)} (given w): Y = U^{P(w)}
        rest = (1 - q) / 9.0
        pick_top = rng.random((k1, n)) < q
        pw = np.where(pick_top, q, rest)
        Yw = rng.random((k1, n)) ** pw
        retained = rng.random((k1, n)) < eps[:, None]
        Y = np.where(retained, Yw, rng.random((k1, n)))
        T1 = (-np.log1p(-Y)).sum(1); p[Z] = stats.gamma.sf(T1, a=n)
    return p, Z

def relaxed_bounds(p, h_fn, ts, alpha=0.05, tmax_point=0.9):
    """Point bounds (grid restricted to t<=tmax_point) and DKW-band simultaneous confidence bounds.
    Returns piL, piU, F, h, piL_lcb, piU_ucb."""
    N = len(p)
    F = np.array([(p <= t).mean() for t in ts])
    h = h_fn(ts)
    m = ts <= tmax_point
    piL = np.max(((F - ts) / (1 - ts))[m])
    piU = np.min(((F - ts) / (h - ts))[m])
    b = np.sqrt(np.log(2 / alpha) / (2 * N))
    piL_lcb = max(np.max((F - b - ts) / (1 - ts)), 0.0)
    piU_ucb = min(np.min((F + b - ts) / (h - ts)), 1.0)
    return piL, min(max(piU, 0), 1), F, h, piL_lcb, piU_ucb

def lp_mech_bounds(F, ts, dictG, band):
    """max/min pi s.t. (1-pi)t + sum_j v_j G_j(t) within band of F(t); sum v_j = pi; v>=0."""
    J = dictG.shape[0]; K = len(ts)
    # variables: [pi, v_1..v_J]
    c = np.zeros(J + 1); c[0] = -1.0
    A_ub = []; b_ub = []
    for k in range(K):
        row = np.zeros(J + 1); row[0] = -ts[k]; row[1:] = dictG[:, k]
        # (1-pi)t + sum v G = t + row·x  <= F+band  ;  >= F-band
        A_ub.append(row); b_ub.append(F[k] + band - ts[k])
        A_ub.append(-row); b_ub.append(-(F[k] - band - ts[k]))
    A_eq = np.zeros((1, J + 1)); A_eq[0, 0] = -1; A_eq[0, 1:] = 1
    bounds = [(0, 1)] + [(0, 1)] * J
    res_max = optimize.linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=A_eq, b_eq=[0], bounds=bounds, method="highs")
    res_min = optimize.linprog(-c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=A_eq, b_eq=[0], bounds=bounds, method="highs")
    return (res_min.x[0] if res_min.success else np.nan, res_max.x[0] if res_max.success else np.nan)

def dict_pvalue_cdfs(n, eps_grid, q_grid, ts, rng, N=20000):
    """Dictionary of p-value CDFs for mechanism members (eps, constant top-prob q, rest spread)."""
    rows = []
    for eps in eps_grid:
        for q in q_grid:
            rest = (1 - q) / 9.0
            pick = rng.random((N, n)) < q
            pw = np.where(pick, q, rest)
            Yw = rng.random((N, n)) ** pw
            ret = rng.random((N, n)) < eps
            Y = np.where(ret, Yw, rng.random((N, n)))
            p = stats.gamma.sf((-np.log1p(-Y)).sum(1), a=n)
            rows.append([(p <= t).mean() for t in ts])
    return np.array(rows)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../figures"); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(1)
    ts = np.concatenate([np.array([0.001, 0.005]), np.linspace(0.01, 0.99, 99)])
    results = {}

    # 1. FFT vs Monte Carlo check of h
    t0 = time.time()
    h_fft, shift = power_envelope(200, 0.3, 0.1, ts)
    h_mc = power_envelope_mc(200, 0.3, 0.1, ts, N=100000, rng=rng)
    i = [np.argmin(abs(ts - v)) for v in (0.01, 0.05, 0.1, 0.5)]
    results["check_h_n200_eps0.3_delta0.1"] = {"t": ts[i].tolist(), "h_fft": h_fft[i].round(4).tolist(), "h_mc": h_mc[i].round(4).tolist(), "per_token_mean_shift": round(float(shift), 4), "sec": round(time.time() - t0, 1)}
    print(json.dumps(results, indent=1))

    # 2. grid: pi_U for true pi = 0.2, heterogeneous DGP (eps ~ Uniform[eps_min, 1], mixed entropy)
    ns = [100, 200, 500, 1000] if not a.quick else [200, 500]
    eps_mins = [0.2, 0.3, 0.5]; deltas = [0.05, 0.1, 0.2]
    N = 20000
    grid = []
    for n in ns:
        for em in eps_mins:
            for d in deltas:
                h_vals, _ = power_envelope(n, em, d, ts)
                hf = lambda tt, hv=h_vals: hv
                p, Z = sample_docs(N, n, 0.2, lambda k, r: r.uniform(em, 1.0, k), "mixed", rng, qmax=min(0.9, 1 - d))
                piL, piU, F, h, lcb, ucb = relaxed_bounds(p, hf, ts)
                grid.append({"n": n, "eps_min": em, "delta": d, "pi_true": 0.2, "pi_L": round(float(piL), 3), "pi_U": round(float(piU), 3), "lcb95": round(float(lcb), 3), "ucb95": round(float(ucb), 3), "h(0.05)": round(float(h[np.argmin(abs(ts-0.05))]), 3)})
                print(grid[-1], flush=True)
    results["grid_pi0.2_hetero"] = grid

    # 3. Does the width depend on the mechanism or only on eps_min? Compare the same eps_min, different (n, delta)
    # already visible in grid. Also: least-favourable DGP attains pi_U (Prop. 2 check)
    n, em, d = 200, 0.3, 0.1
    h_vals, _ = power_envelope(n, em, d, ts)
    # least favourable docs: eps = eps_min, P = P* (two-point: q=1-delta, other mass delta on ONE token)
    Nlf = 20000; Z = rng.random(Nlf) < 0.2; p = np.empty(Nlf)
    p[~Z] = stats.gamma.sf(rng.gamma(n, 1.0, (~Z).sum()), a=n)
    k1 = Z.sum(); U = rng.random((k1, n)); B = rng.random((k1, n)) < em; C = rng.random((k1, n)) < (1 - d)
    Y = np.where(B, np.where(C, U ** (1 - d), U ** d), U)
    p[Z] = stats.gamma.sf((-np.log1p(-Y)).sum(1), a=n)
    piL, piU, F, h, lcb, ucb = relaxed_bounds(p, lambda tt: h_vals, ts)
    results["least_favourable_attains"] = {"n": n, "eps_min": em, "delta": d, "pi_true": 0.2, "pi_L": round(float(piL), 3), "pi_U": round(float(piU), 3), "lcb95": round(float(lcb), 3), "ucb95": round(float(ucb), 3)}
    print(results["least_favourable_attains"])

    # 4. LP mechanism-class bounds vs relaxed pi_U on the heterogeneous DGP (n=200, eps_min=0.3, delta=0.1)
    p, Z = sample_docs(N, n, 0.2, lambda k, r: r.uniform(em, 1.0, k), "mixed", rng, qmax=min(0.9, 1 - d))
    piL, piU, F, h, lcb, ucb = relaxed_bounds(p, lambda tt: h_vals, ts)
    tsub = ts[::4]
    Fsub = np.array([(p <= t).mean() for t in tsub])
    dictG = dict_pvalue_cdfs(n, eps_grid=[0.3, 0.4, 0.5, 0.6, 0.8, 1.0], q_grid=[0.3, 0.5, 0.7, 0.85, 0.9], ts=tsub, rng=rng, N=8000)
    band = np.sqrt(np.log(2 / 0.05) / (2 * N))  # DKW 95%
    lo, hi = lp_mech_bounds(Fsub, tsub, dictG, band)
    results["lp_vs_relaxed_hetero"] = {"n": n, "eps_min": em, "delta": d, "pi_true": 0.2, "relaxed_pi_L": round(float(piL), 3), "relaxed_pi_U": round(float(piU), 3), "relaxed_lcb95": round(float(lcb), 3), "relaxed_ucb95": round(float(ucb), 3), "lp_mech_lo": round(float(lo), 3), "lp_mech_hi": round(float(hi), 3), "dkw_band": round(float(band), 4)}
    print(results["lp_vs_relaxed_hetero"])

    with open(os.path.join(a.out, "go_no_go_results.json"), "w") as f:
        json.dump(results, f, indent=1)

    # figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
        for ax, d in zip(axes, deltas):
            for em in eps_mins:
                ys = [g["ucb95"] for g in grid if g["delta"] == d and g["eps_min"] == em]
                xs = [g["n"] for g in grid if g["delta"] == d and g["eps_min"] == em]
                ax.plot(xs, ys, marker="o", label=f"eps_min={em}")
            ax.axhline(0.2, ls="--", c="k", lw=0.8); ax.set_xscale("log"); ax.set_title(f"delta={d} (max NTP prob <= {1-d})"); ax.set_xlabel("doc length n")
        axes[0].set_ylabel("95% upper conf. bound on pi (true 0.2, N=20000 docs)"); axes[0].legend()
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "pi_U_grid.png"), dpi=150)
    except Exception as e:
        print("plot skipped:", e)

if __name__ == "__main__":
    main()
