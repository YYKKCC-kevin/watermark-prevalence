"""Stage 4 analyses (responses to referees).  For one (model, scheme):
 1. class check over ALL thresholds used by inference, with a simultaneous lower confidence bound on the gap
    min_t (G_s(t) - h_s(t)) (DKW band, Bonferroni over strata);
 2. coverage under the sampling design of Corollary 1: labels iid Bernoulli(pi), watermarked documents drawn WITH replacement
    from each pool, human p-values exactly U(0,1) (resampling the finite human pool would treat its own sampling noise as a
    population departure from uniformity), target pi_cond = sum_s (N_s/N) pi_s with pi_s computed from the pool stratum shares; Clopper-Pearson CIs;
 3. population upper end (plug-in with the pool's G and the true pi) next to the mean reported upper bound;
 4. one-threshold comparator (detection count at t0=0.05 with an exact binomial CI and sensitivity range [h(t0),1],
    i.e. a Rogan-Gladen-type correction with worst-case sensitivity);
 5. substitution pools evaluated with sigma = 0 (envelope ignoring substitutions) and with a substitution budget sigma;
 6. realized-fraction confidence sequence at t = 0.05 and t = 0.3 on the drifting stream.
Usage: python3 analysis_stage4.py --tag opt-1p3b --scheme gumbel
"""
import argparse, json, os, numpy as np
from scipy import stats
from analysis import envelope, TS, BINS, strat, bounds_corpus
from betting_cs import cs_running_fraction

ALPHA = 0.05; T0 = 0.05

def load(path, nmin=50):
    e = np.load(path, allow_pickle=True)
    n = e["n"].astype(int) if "n" in e.files else np.array([len(x) for x in e["pivots"]])
    p = e["pvals"].astype(float); m = n >= nmin
    return p[m], n[m], {k: e[k][m] for k in ("eps_eff", "sig_eff") if k in e.files}

def strata_of(n): return np.array([strat(x) for x in n])

def cp(k, n, a):
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi

def one_threshold(p, n, hls, alpha=ALPHA):
    s = strata_of(n); K = len(BINS); L = U = 0.0; it0 = int(np.argmin(abs(TS - T0)))
    for k in range(K):
        m = s == k; w = m.mean()
        if m.sum() < 30: lo_, hi_ = 0.0, 1.0
        else:
            Flo, Fhi = cp(int((p[m] <= T0).sum()), int(m.sum()), alpha / K); h0 = hls[k][it0]
            lo_ = max(0.0, (Flo - T0) / (1 - T0)); hi_ = min(1.0, (Fhi - T0) / (h0 - T0)) if h0 > T0 else 1.0
        L += w * lo_; U += w * hi_
    return L, U

def class_check(p, n, hls, alpha=ALPHA):
    s = strata_of(n); out = []
    for k in range(len(BINS)):
        m = s == k
        if m.sum() < 30: out.append(None); continue
        G = np.array([(p[m] <= t).mean() for t in TS]); b = np.sqrt(np.log(2 * len(BINS) / alpha) / (2 * m.sum()))
        gap = G - hls[k]; ok = hls[k] - TS > 0.02
        ratio = (G[ok] - TS[ok]) / (hls[k][ok] - TS[ok]); ratio_lcb = (G[ok] - b - TS[ok]) / (hls[k][ok] - TS[ok])
        vac = not ok.any()   # envelope never exceeds t by 0.02: the class constraint is vacuous
        out.append({"N": int(m.sum()), "min_ratio": None if vac else round(float(ratio.min()), 3), "t_at_min_ratio": None if vac else float(TS[ok][np.argmin(ratio)]),
                    "min_ratio_lcb": None if vac else round(float(ratio_lcb.min()), 3),
                    "min_gap": round(float(gap.min()), 3), "t_at_min": float(TS[np.argmin(gap)]),
                                      "min_gap_lcb": round(float((gap - b).min()), 3), "min_gap_t_le_0.5": round(float(gap[TS <= 0.5].min()), 3)})
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--scheme", required=True)
    ap.add_argument("--gen", default="data/gen"); ap.add_argument("--out", default="data/results"); ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--N", type=int, default=4000); ap.add_argument("--eps_min", type=float, default=0.3); ap.add_argument("--sigmas", type=str, default="0.05,0.15"); ap.add_argument("--null", default="uniform", choices=["uniform", "pool"]); ap.add_argument("--extra_pools", type=str, default="", help="comma list name=file_suffix; if set, ONLY these pools are analysed")
    a = ap.parse_args(); rng = np.random.default_rng(4242); base = f"{a.gen}/{a.tag}_{a.scheme}"; SIG = [float(x) for x in a.sigmas.split(",")]
    hp, hn, _ = load(base + "_human.npz")
    if a.extra_pools:
        pools = {kv.split("=")[0]: load(base + kv.split("=")[1] + ".npz") for kv in a.extra_pools.split(",")}
    else:
        pools = {"scattered": load(base + "_edit_scattered_eps0.3.npz"), "span": load(base + "_edit_span_eps0.3.npz"),
                 "subst": load(base + "_subst.npz"), "paraphrase": load(base + "_paraphrase.npz")}
    res = {"tag": a.tag, "scheme": a.scheme, "eps_min": a.eps_min, "sigma_budgets": SIG, "R": a.R, "N": a.N, "design": "iid labels, watermarked docs sampled with replacement from the pool, target pi_cond; human p-values exactly U(0,1) (Assumption 1) with human lengths from the pool unless --null pool", "null": a.null}
    if "subst" in pools and "sig_eff" in pools["subst"][2]:
        se = pools["subst"][2]["sig_eff"]; ee = pools["subst"][2]["eps_eff"]
        res["subst_pool"] = {"docs": int(len(se)), "sig_eff_mean": round(float(np.nanmean(se)), 3), "sig_eff_q95": round(float(np.nanquantile(se, .95)), 3),
                             "frac_sig_eff_gt_budget": {str(s): round(float(np.nanmean(se > s)), 3) for s in SIG}, "eps_eff_mean": round(float(np.nanmean(ee)), 3),
                             "frac_eps_eff_lt_eps_min": round(float(np.nanmean(ee < a.eps_min)), 3), "detect_rate": round(float((pools["subst"][0] <= 0.05).mean()), 3)}
    env = {}
    for d in (0.1, 0.2):
        for sg in [0.0] + SIG:
            env[(d, sg)] = [envelope(a.scheme, lo, a.eps_min, d, sigma=sg) for (lo, hi) in BINS]
    hs = strata_of(hn); res["class_check"] = {}; res["coverage"] = []
    for pool, (pv, nv, extra) in pools.items():
        sigs = [0.0] + SIG if pool == "subst" else [0.0]
        for d in (0.1, 0.2):
            for sg in sigs:
                res["class_check"][f"{pool}_d{d}_s{sg}"] = class_check(pv, nv, env[(d, sg)])
        ws = strata_of(nv); Pw = np.array([(ws == k).mean() for k in range(len(BINS))]); Ph = np.array([(hs == k).mean() for k in range(len(BINS))])
        for pi in (0.05, 0.2, 0.5):
            for d in (0.1, 0.2):
                for sg in sigs:
                    hls = env[(d, sg)]; cov = cov1 = 0; Ls, Us, L1s, U1s, tg = [], [], [], [], []
                    for r in range(a.R):
                        Z = rng.random(a.N) < pi; nz = int(Z.sum())
                        iw = rng.integers(0, len(pv), nz); ih = rng.integers(0, len(hp), a.N - nz)
                        p = np.concatenate([pv[iw], rng.random(a.N - nz) if a.null == 'uniform' else hp[ih]]); n = np.concatenate([nv[iw], hn[ih]])
                        s = strata_of(n); Ns = np.array([(s == k).sum() for k in range(len(BINS))])
                        pis = np.where(pi * Pw + (1 - pi) * Ph > 0, pi * Pw / np.maximum(pi * Pw + (1 - pi) * Ph, 1e-300), 0)
                        target = float((Ns / a.N * pis).sum())
                        L, U, _ = bounds_corpus(p, n, hls, ALPHA); L1, U1 = one_threshold(p, n, hls)
                        cov += (L <= target <= U); cov1 += (L1 <= target <= U1); Ls.append(L); Us.append(U); L1s.append(L1); U1s.append(U1); tg.append(target)
                    # population upper end with the pool's G and the true mixture (no sampling noise)
                    Gs = [np.array([(pv[ws == k] <= t).mean() for t in TS]) if (ws == k).sum() else None for k in range(len(BINS))]
                    popU = 0.0
                    for k in range(len(BINS)):
                        wk = pi * Pw[k] + (1 - pi) * Ph[k]
                        if wk == 0: continue
                        pik = pis[k]
                        if Gs[k] is None or pik == 0: continue
                        F = (1 - pik) * TS + pik * Gs[k]; ok = hls[k] > TS
                        popU += wk * (min(1.0, float(np.min((F[ok] - TS[ok]) / (hls[k][ok] - TS[ok])))) if ok.any() else 1.0)
                    lo, hi = cp(cov, a.R, 0.05)
                    res["coverage"].append({"pool": pool, "pi": pi, "delta": d, "sigma": sg, "coverage": round(cov / a.R, 3), "coverage_CI95": [round(lo, 3), round(hi, 3)],
                                            "mean_target": round(float(np.mean(tg)), 4), "mean_L": round(float(np.mean(Ls)), 4), "mean_U": round(float(np.mean(Us)), 4),
                                            "population_U": round(popU, 4), "one_threshold_coverage": round(cov1 / a.R, 3),
                                            "one_threshold_mean_L": round(float(np.mean(L1s)), 4), "one_threshold_mean_U": round(float(np.mean(U1s)), 4)})
                    print(res["coverage"][-1], flush=True)
    if a.extra_pools:
        json.dump(res, open(f"{a.out}/{a.tag}_{a.scheme}_stage4_lowpower" + ("" if a.null == "uniform" else "_poolnull") + ".json", "w"), indent=1); return
    # stream at t=0.05 and t=0.3 (scattered pool, n>=100, envelope at n=100, delta=0.2, sigma=0)
    pv, nv, _ = pools["scattered"]; mw = nv >= 100; mh = hn >= 100; hl = envelope(a.scheme, 100, a.eps_min, 0.2)
    prob = np.concatenate([np.zeros(1000), np.full(1000, 0.4), np.full(1000, 0.1)]); Z = rng.random(len(prob)) < prob
    p = np.empty(len(Z)); p[Z] = rng.permutation(pv[mw])[:Z.sum()]; p[~Z] = rng.permutation(hp[mh])[:(~Z).sum()]
    pibar = np.cumsum(Z) / np.arange(1, len(Z) + 1); res["stream"] = {}
    for t in (0.05, 0.3):
        it = int(np.argmin(abs(TS - t))); L, U = cs_running_fraction(p, TS[it], hl[it], ALPHA)
        res["stream"][str(t)] = {"a": round(float(hl[it] - TS[it]), 4), "covered_all_k": bool(np.all((L <= pibar) & (pibar <= U))),
                                 "at_k2000": [round(float(L[1999]), 3), round(float(U[1999]), 3), round(float(pibar[1999]), 3)],
                                 "at_k3000": [round(float(L[-1]), 3), round(float(U[-1]), 3), round(float(pibar[-1]), 3)]}
    print(res["stream"])
    json.dump(res, open(f"{a.out}/{a.tag}_{a.scheme}_stage4" + ("" if a.null == "uniform" else "_poolnull") + ".json", "w"), indent=1)

if __name__ == "__main__":
    main()
