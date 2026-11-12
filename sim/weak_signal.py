"""Weak-signal simulations with a known continuous population (must-do item 2; Codex review 2026-11-12).
Watermarked documents follow the least-favourable mechanism (retention eps = eps_min = 0.3, NTP P* = (1-delta, delta)),
so their p-value law equals the envelope h and detection power at t = 0.05 is 0.18-0.85 depending on (n, delta).
Extra settings: substitutions near the strictness boundary, and a mixed weak/strong population.
Design A (realized fraction): exactly round(pi N) watermarked documents at random positions; compare
  multi  : realized_bounds over a pre-specified grid of 12 thresholds (union bound)
  t005   : single threshold t = 0.05
  t_env  : single threshold chosen from the envelope alone (envelope_chosen_threshold)
  t_hmax : single threshold maximising h(t) - t (data-free)
  t_cal  : single threshold chosen on a fresh independent pilot corpus in every replication (min upper bound), then applied
Design B (iid labels, Corollary 1): DKW band inversion, checked against the population identified set
  (upper end pi_U = inf_t (F - t)/(h - t) and Storey-type lower end, computed from the known population CDF).
Human p-values are exactly uniform.  Run: python3 -I weak_signal.py --out ../figures [--R 200]
"""
import argparse, json, os, sys, time
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certified import certified_envelope, band_bounds
from realized import realized_bounds, envelope_chosen_threshold

TS_GRID = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
TS_FINE = np.concatenate([np.array([0.001, 0.005]), np.linspace(0.01, 0.99, 99)])

def wm_pvalues(k, n, delta, rng, eps=0.3, sigma=0.0, mixed=False):
    """Least-favourable documents; with sigma > 0 a share sigma of positions are worst-case substitutions
    (pivot a^q V, q = 1 - delta); mixed=True makes half the documents strong (eps ~ U[0.3,1], q ~ U[0.5,0.8], spread)."""
    q = np.full((k, n), 1 - delta); rest = np.full((k, n), delta); e = np.full(k, eps)
    if mixed:
        strong = rng.random(k) < 0.5
        qs = rng.uniform(0.5, 0.8, (k, n)); q = np.where(strong[:, None], qs, q); rest = np.where(strong[:, None], (1 - qs) / 9, rest)
        e = np.where(strong, rng.uniform(0.3, 1.0, k), e)
    pick = rng.random((k, n)) < q; pw = np.where(pick, q, rest)
    Yw = rng.random((k, n)) ** pw
    u = rng.random((k, n)); ret = u < e[:, None]; sub = (u >= e[:, None]) & (u < e[:, None] + sigma)
    Ysub = rng.random((k, n)) ** (1 - delta) * rng.random((k, n))
    Y = np.where(ret, Yw, np.where(sub, Ysub, rng.random((k, n))))
    return stats.gamma.sf((-np.log1p(-np.clip(Y, 0, 1 - 1e-12))).sum(1), a=n)

def single_bounds(p, ti, hl, alpha):
    return realized_bounds(p, np.zeros(len(p), int), 1, TS_GRID[[ti]], [hl[[ti]]], alpha)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../figures"); ap.add_argument("--R", type=int, default=200)
    a = ap.parse_args(); rng = np.random.default_rng(20261112); alpha = 0.05
    configs = [  # name, n, delta (true = assumed), sigma (true), sigma_max (assumed), mixed
        ("LF d=0.05 n=400", 400, 0.05, 0.0, 0.0, False), ("LF d=0.2 n=50", 50, 0.2, 0.0, 0.0, False),
        ("LF d=0.1 n=400", 400, 0.1, 0.0, 0.0, False), ("LF d=0.2 n=200", 200, 0.2, 0.0, 0.0, False),
        ("LF d=0.2 n=400", 400, 0.2, 0.0, 0.0, False), ("LF d=0.2 n=800", 800, 0.2, 0.0, 0.0, False),
        ("LF+subst d=0.2 n=200 s=.07", 200, 0.2, 0.07, 0.075, False), ("mixed d=0.2 n=200", 200, 0.2, 0.0, 0.0, True)]
    rows = []; t0 = time.time()
    for (name, n, d, sig, sigmax, mixed) in configs:
        hl = certified_envelope(n, 0.3, d, TS_GRID, sigma=sigmax); hlf = certified_envelope(n, 0.3, d, TS_FINE, sigma=sigmax)
        sens = float((wm_pvalues(20000, n, d, rng, sigma=sig, mixed=mixed) <= 0.05).mean())
        t_env = envelope_chosen_threshold(TS_GRID, hl)
        # population CDF of watermarked p-values on the fine grid (large Monte Carlo; for pure LF it equals h up to MC error)
        pw_big = wm_pvalues(100000, n, d, rng, sigma=sig, mixed=mixed); Gf = np.array([(pw_big <= t).mean() for t in TS_FINE])
        for N in (500, 2000):
            for pi in (0.05, 0.2):
                K = int(round(pi * N)); res = {k: [] for k in ("multi", "t005", "t_env", "t_hmax", "t_cal")}; covB = 0; covBset = 0; LB = []; UB = []
                t_hmax = int(np.argmax(np.where(hl > TS_GRID, hl - TS_GRID, -1))); tcal_choices = []
                Fpop = (1 - pi) * TS_FINE + pi * Gf; okf = hlf > TS_FINE
                piU_pop = min(1.0, float(np.min((Fpop[okf] - TS_FINE[okf]) / (hlf[okf] - TS_FINE[okf]))))
                piL_pop = max(0.0, float(np.max((Fpop - TS_FINE) / (1 - TS_FINE))))
                for r in range(a.R):
                    Z = np.zeros(N, bool); Z[rng.choice(N, K, replace=False)] = True
                    p = np.empty(N); p[Z] = wm_pvalues(K, n, d, rng, sigma=sig, mixed=mixed); p[~Z] = rng.random(N - K)
                    target = K / N; strat = np.zeros(N, int)
                    # a fresh, independent pilot corpus in every replication (Codex round 3: stability of the pilot choice)
                    Zp = np.zeros(N, bool); Zp[rng.choice(N, K, replace=False)] = True
                    pp = np.empty(N); pp[Zp] = wm_pvalues(K, n, d, rng, sigma=sig, mixed=mixed); pp[~Zp] = rng.random(N - K)
                    t_cal = int(np.argmin([single_bounds(pp, j, hl, alpha)[1] for j in range(len(TS_GRID))])); tcal_choices.append(float(TS_GRID[t_cal]))
                    for key, (L, U) in (("multi", realized_bounds(p, strat, 1, TS_GRID, [hl], alpha)), ("t005", single_bounds(p, 2, hl, alpha)),
                                        ("t_env", single_bounds(p, t_env, hl, alpha)), ("t_hmax", single_bounds(p, t_hmax, hl, alpha)),
                                        ("t_cal", single_bounds(p, t_cal, hl, alpha))):
                        res[key].append((L, U, L <= target <= U))
                    # design B: iid labels, Corollary 1 band inversion against the population identified set
                    Zb = rng.random(N) < pi; kb = int(Zb.sum()); pb = np.empty(N); pb[Zb] = wm_pvalues(kb, n, d, rng, sigma=sig, mixed=mixed); pb[~Zb] = rng.random(N - kb)
                    Lb, Ub = band_bounds(pb, hlf, TS_FINE, alpha); LB.append(Lb); UB.append(Ub)
                    covB += (Lb <= pi <= Ub); covBset += (Lb <= piL_pop + 1e-9) and (Ub >= piU_pop - 1e-9)
                row = {"config": name, "n": n, "delta": d, "sigma_true": sig, "sigma_max": sigmax, "sens_t005": round(sens, 3), "N": N, "pi": pi, "R": a.R,
                       "t_env_value": float(TS_GRID[t_env]), "t_hmax_value": float(TS_GRID[t_hmax]), "t_cal_choices": {str(v): tcal_choices.count(v) for v in sorted(set(tcal_choices))}, "pop_identified_set": [round(piL_pop, 4), round(piU_pop, 4)],
                       "designB_cov_pi": covB / a.R, "designB_cov_whole_set": covBset / a.R, "designB_mean": [round(float(np.mean(LB)), 4), round(float(np.mean(UB)), 4)]}
                for k, v in res.items():
                    v = np.array(v, float); row[k] = {"coverage": float(v[:, 2].mean()), "mean_L": round(float(v[:, 0].mean()), 4), "mean_U": round(float(v[:, 1].mean()), 4),
                                                      "U_q10_q90": [round(float(np.quantile(v[:, 1], 0.1)), 4), round(float(np.quantile(v[:, 1], 0.9)), 4)]}
                rows.append(row); print(json.dumps(row), f"{time.time()-t0:.0f}s", flush=True)
    json.dump(rows, open(os.path.join(a.out, "weak_signal_results.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
