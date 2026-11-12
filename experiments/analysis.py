"""Main real-data analysis for one (model, scheme): partial-identification bounds, baselines, sequential CS, (S1) check.

Pools (data/gen/<tag>_<scheme>*.npz): human nulls; watermarked unedited; edited (scattered / span, eps_i~U[0.3,1]); paraphrased.
Length strata: n in [50,100), [100,200), [200, inf).  Envelope per stratum uses n_min of the stratum (h is increasing in n,
checked numerically), assumed eps_min = 0.3 and delta in {0.1, 0.2}; Gumbel: certified h_lower (sim/certified.py);
KGW: closed form h_kgw (sim/kgw.py).  Bounds: DKW band per stratum at level alpha/K (Bonferroni over K strata),
aggregated with the observed stratum weights.  Coverage over R random corpus draws for pi in {0.05, 0.2, 0.5}.
Baselines: naive frac(p<=0.05); Storey lcb; reference-sample point estimate using the UNEDITED pool as reference.
Sequential: running-fraction CS (Thm 3) on a drifting stream.
Usage: PYTHONPATH=...:sim python3 analysis.py --tag Qwen3-1p7B --scheme gumbel
"""
import argparse, json, os, numpy as np
from scipy import stats
from certified import certified_envelope, band_bounds
from kgw import h_kgw
from betting_cs import cs_running_fraction

BINS = [(50, 100), (100, 200), (200, 10**9)]
TS = np.concatenate([[0.001, 0.005], np.linspace(0.01, 0.99, 99)])

def envelope(scheme, n, eps_min, delta, sigma=0.0):
    if scheme == "gumbel": return certified_envelope(n, eps_min, delta, TS, sigma=sigma)
    h, _ = h_kgw(TS, n, 0.5, eps_min, delta, 2.0, V=None, sigma=sigma); return np.maximum(h - 1e-9, 0)  # V=None: p10=gamma(1-gamma), conservative for every vocabulary

def strat(n): return next(i for i, (lo, hi) in enumerate(BINS) if lo <= n < hi)

def bounds_corpus(p, n, hls, alpha=0.05):
    """Aggregated [L, U] from per-stratum DKW bands (Bonferroni alpha/K) weighted by observed stratum frequencies."""
    s = np.array([strat(x) for x in n]); K = len(BINS); L = U = 0.0; per = []
    for k in range(K):
        m = s == k; w = m.mean()
        if m.sum() < 30: lcb, ucb = 0.0, 1.0
        else: lcb, ucb = band_bounds(p[m], hls[k], TS, alpha / K)
        L += w * lcb; U += w * ucb; per.append((int(m.sum()), round(lcb, 4), round(ucb, 4)))
    return L, U, per

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--scheme", required=True)
    ap.add_argument("--gen", default="data/gen"); ap.add_argument("--out", default="data/results"); ap.add_argument("--R", type=int, default=50)
    ap.add_argument("--N", type=int, default=4000); ap.add_argument("--eps_min", type=float, default=0.3); ap.add_argument("--edit_eps_lo", type=float, default=0.3)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(2026)
    base = f"{a.gen}/{a.tag}_{a.scheme}"
    wm = np.load(base + ".npz", allow_pickle=True); hu = np.load(base + "_human.npz", allow_pickle=True)
    pools = {"unedited": (wm["pvals"].astype(float), np.array([len(x) for x in wm["pivots"]]), None)}
    for mode in ("scattered", "span"):
        f = f"{base}_edit_{mode}_eps{a.edit_eps_lo}.npz"
        if os.path.exists(f): e = np.load(f, allow_pickle=True); pools[mode] = (e["pvals"].astype(float), e["n"].astype(int), e["eps_eff"].astype(float))
    if os.path.exists(base + "_paraphrase.npz"): e = np.load(base + "_paraphrase.npz", allow_pickle=True); pools["paraphrase"] = (e["pvals"].astype(float), e["n"].astype(int), None)
    hp, hn = hu["pvals"].astype(float), hu["n"].astype(int)
    def keep(pv, n, ee):   # drop documents with fewer than 50 scored pivots (below the first stratum)
        m = n >= BINS[0][0]; return (pv[m], n[m], None if ee is None else ee[m])
    pools = {k: keep(*v) for k, v in pools.items()}; hp, hn = hp[hn >= BINS[0][0]], hn[hn >= BINS[0][0]]
    res = {"tag": a.tag, "scheme": a.scheme, "edit_eps_lo": a.edit_eps_lo, "assumed_eps_min": a.eps_min, "human_report": str(hu["report"]), "n_human": int(len(hp)), "n_wm": int(len(pools["unedited"][0]))}

    # (S1): max-prob of the unwatermarked NTP. Definition 1 constrains SCORED positions only (first occurrence of each
    # h-gram context within a document, positions >= h), so report shares on scored positions; all-position shares and
    # the generated-text dedup rate are kept for reference (Stage 2.5 integrity fix, 2026-11-12).
    hh = int(wm["h"]); mp_sc, mp_all, n_tot, n_kept = [], [], 0, 0
    for ids, mpx, pl in zip(wm["ids"], wm["maxp"], wm["prompt_len"]):
        ids = [int(x) for x in ids]; mpx = np.asarray(mpx, float); seen = set(); mp_all.append(mpx)
        for t in range(max(int(pl), hh), min(len(ids), int(pl) + len(mpx))):
            ctx = tuple(ids[t - hh:t]); n_tot += 1
            if ctx in seen: continue
            seen.add(ctx); n_kept += 1; mp_sc.append(mpx[t - int(pl)])
    mp = np.array(mp_sc); mp_allc = np.concatenate(mp_all); docmed = np.array([np.median(np.asarray(x, float)) for x in wm["maxp"]])
    res["S1_maxprob_all_positions"] = {"frac_pos_gt_0.9": round(float((mp_allc > 0.9).mean()), 3), "frac_pos_gt_0.99": round(float((mp_allc > 0.99).mean()), 3)}
    res["gen_dedup_drop_frac"] = round(1 - n_kept / n_tot, 3)
    res["S1_maxprob"] = {"pos_quantiles_10_50_90_99": np.quantile(mp, [.1, .5, .9, .99]).round(3).tolist(),
                         "frac_pos_gt_0.8": round(float((mp > 0.8).mean()), 3), "frac_pos_gt_0.9": round(float((mp > 0.9).mean()), 3), "frac_pos_gt_0.99": round(float((mp > 0.99).mean()), 3),
                         "doc_median_maxp_quantiles_10_50_90": np.quantile(docmed, [.1, .5, .9]).round(3).tolist()}
    res["length"] = {"wm_n_quantiles_10_50_90": np.quantile(pools["unedited"][1], [.1, .5, .9]).tolist(), "human_n_quantiles_10_50_90": np.quantile(hn, [.1, .5, .9]).tolist()}
    for k, (pv, n, ee) in pools.items():
        r = {"docs": int(len(pv)), "frac_p_le_0.05": round(float((pv <= 0.05).mean()), 3), "frac_p_le_0.01": round(float((pv <= 0.01).mean()), 3)}
        if ee is not None: r.update({"eps_eff_mean": round(float(np.nanmean(ee)), 3), "eps_eff_q05": round(float(np.nanquantile(ee, .05)), 3), "frac_eps_eff_below_eps_min": round(float(np.nanmean(ee < a.eps_min)), 3)})
        res[f"pool_{k}"] = r

    # envelopes per stratum, monotonicity check
    hls = {}; mono = {}
    for d in (0.1, 0.2):
        hls[d] = [envelope(a.scheme, lo, a.eps_min, d) for lo, hi in BINS]
        mono[d] = [bool((envelope(a.scheme, lo, a.eps_min, d) <= envelope(a.scheme, min(hi, 330), a.eps_min, d) + 1e-9).all()) for lo, hi in BINS]
    res["envelope_monotone_in_n"] = mono

    # class check: empirical G_s(t) of each watermarked pool vs h_lower per stratum (min over t of G - h, t<=0.5)
    cc = {}
    for k, (pv, n, ee) in pools.items():
        s = np.array([strat(x) for x in n]); row = {}
        for d in (0.1, 0.2):
            gaps = []
            for j in range(len(BINS)):
                m = s == j
                if m.sum() < 30: gaps.append(None); continue
                G = np.array([(pv[m] <= t).mean() for t in TS]); sel = TS <= 0.5
                gaps.append(round(float((G - hls[d][j])[sel].min()), 3))
            row[f"delta{d}_min_gap_G_minus_h_by_stratum"] = gaps
        cc[k] = row
    res["class_check"] = cc

    # mixture draws
    mix = []
    for pool in [k for k in ("scattered", "span", "paraphrase") if k in pools]:
        pv_w, n_w, _ = pools[pool]; pv_u, n_u, _ = pools["unedited"]
        for pi in (0.05, 0.2, 0.5):
            nw = int(round(pi * a.N)); nh = a.N - nw
            if nw > len(pv_w) or nh > len(hp): continue
            cov = {0.1: 0, 0.2: 0}; Ls = {0.1: [], 0.2: []}; Us = {0.1: [], 0.2: []}; naive = []; storey = []; ref = []
            for r in range(a.R):
                iw = rng.choice(len(pv_w), nw, replace=False); ih = rng.choice(len(hp), nh, replace=False)
                p = np.concatenate([pv_w[iw], hp[ih]]); n = np.concatenate([n_w[iw], hn[ih]])
                for d in (0.1, 0.2):
                    L, U, per = bounds_corpus(p, n, hls[d]); cov[d] += (L <= pi <= U); Ls[d].append(L); Us[d].append(U)
                F05 = (p <= 0.05).mean(); naive.append(F05); storey.append(max((F05 - 0.05) / 0.95, 0))
                Gref = (pv_u <= 0.05).mean(); ref.append((F05 - 0.05) / (Gref - 0.05))
            mix.append({"pool": pool, "pi": pi, "N": a.N, "R": a.R,
                        **{f"delta{d}": {"coverage": cov[d] / a.R, "mean_L": round(float(np.mean(Ls[d])), 4), "mean_U": round(float(np.mean(Us[d])), 4)} for d in (0.1, 0.2)},
                        "naive_frac_p_le_0.05_mean": round(float(np.mean(naive)), 4), "storey_lcb_point_mean": round(float(np.mean(storey)), 4),
                        "reference_noedit_point_mean": round(float(np.mean(ref)), 4)})
            print(mix[-1], flush=True)
    res["mixtures"] = mix

    # sequential running-fraction CS on a drifting stream (docs with n>=100 only; envelope at n=100)
    if "scattered" in pools:
        pv_w, n_w, _ = pools["scattered"]; mw = n_w >= 100; mh = hn >= 100
        hl = envelope(a.scheme, 100, a.eps_min, 0.2); t = 0.05; hl_t = hl[np.argmin(abs(TS - t))]
        prob = np.concatenate([np.zeros(1000), np.full(1000, 0.4), np.full(1000, 0.1)])  # 3000 docs: fits the ~3.8k human docs with n>=100
        Z = rng.random(len(prob)) < prob
        need_w, need_h = int(Z.sum()), int((~Z).sum())
        if need_w <= mw.sum() and need_h <= mh.sum():
            pw = rng.permutation(pv_w[mw])[:need_w]; ph = rng.permutation(hp[mh])[:need_h]
            p = np.empty(len(Z)); p[Z] = pw; p[~Z] = ph
            L, U = cs_running_fraction(p, t, hl_t, 0.05); k = np.arange(1, len(Z) + 1); pibar = np.cumsum(Z) / k
            cps = [100, 250, 500, 1000, 1250, 1500, 2000, 2500, 3000]
            res["sequential"] = {"pattern": "0 (1000) -> 0.4 (1000) -> 0.1 (1000)", "assumed": [a.eps_min, 0.2], "t": t, "h_lower_t_n100": round(float(hl_t), 4),
                                 "covered_all_k": bool(np.all((L <= pibar) & (pibar <= U))), "checkpoints": cps,
                                 "L": L[np.array(cps) - 1].round(3).tolist(), "U": U[np.array(cps) - 1].round(3).tolist(), "pibar": pibar[np.array(cps) - 1].round(3).tolist()}
            np.savez(f"{a.out}/{a.tag}_{a.scheme}_editlo{a.edit_eps_lo}_sequential.npz", L=L, U=U, pibar=pibar)
        else:
            res["sequential"] = {"skipped": f"need wm {need_w} / human {need_h}, have {int(mw.sum())} / {int(mh.sum())}"}
    with open(f"{a.out}/{a.tag}_{a.scheme}_editlo{a.edit_eps_lo}_analysis.json", "w") as f: json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k not in ("mixtures",)}, indent=1, default=float))

if __name__ == "__main__":
    main()
