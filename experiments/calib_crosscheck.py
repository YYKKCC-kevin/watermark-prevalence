"""Cross-checks requested in Codex round 3 (2026-11-12), for one (model, scheme) under the original key 20261111:
 (a) old human batch (C4 eligible docs 2600-6599, n_docs 4000) vs new batch (6600-18599, 12000): difference of empirical
     CDFs at the 12 thresholds with its standard error, overall and per stratum;
 (b) OLD design (iid labels, humans resampled WITH replacement from a pool, DKW band inversion of Corollary 1 at N=4000,
     target pi_cond) run on the OLD pool and on the NEW pool;
 (c) NEW bound (Proposition R1, realized share, N=2000, humans WITHOUT replacement) on the OLD pool and on the NEW pool;
 (d) Proposition R2 with 5 independent reference/audit splits of the new pool (reference 4000), 100 corpora per split.
Watermarked documents: the scattered-edit pool of the original key.  delta = 0.2, pi in {0.05, 0.2}.
Usage: PYTHONPATH=.:../sim python3 calib_crosscheck.py --tag Qwen3-1p7B --scheme kgw"""
import argparse, json, numpy as np
from scipy import stats
from analysis import envelope, BINS, strat, TS, bounds_corpus
from realized import realized_bounds, null_band_from_reference
from calib_analysis import load, TS_GRID, cp

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--scheme", required=True)
    ap.add_argument("--R", type=int, default=200); ap.add_argument("--out", default="data/results")
    a = ap.parse_args(); rng = np.random.default_rng(31337); K = len(BINS); res = {"tag": a.tag, "scheme": a.scheme}
    old = load(f"data/gen/{a.tag}_{a.scheme}_human.npz"); new = load(f"data/calib/k20261111/{a.tag}_{a.scheme}_human.npz")
    wp, wn = load(f"data/gen/{a.tag}_{a.scheme}_edit_scattered_eps0.3.npz")
    # (a) batch comparison
    Fo = np.array([(old[0] <= t).mean() for t in TS_GRID]); Fn = np.array([(new[0] <= t).mean() for t in TS_GRID])
    se = np.sqrt(TS_GRID * (1 - TS_GRID) * (1 / len(old[0]) + 1 / len(new[0])))
    res["batch_diff"] = {"old_minus_new": (Fo - Fn).round(4).tolist(), "se": se.round(4).tolist(), "max_abs_z": round(float(np.max(np.abs(Fo - Fn) / se)), 2),
                         "ks_2samp": [round(float(x), 4) for x in stats.ks_2samp(old[0], new[0])], "ks_old": [round(float(x), 4) for x in stats.kstest(old[0], "uniform")],
                         "ks_new": [round(float(x), 4) for x in stats.kstest(new[0], "uniform")], "n_old": int(len(old[0])), "n_new": int(len(new[0]))}
    hls_full = [envelope(a.scheme, lo, 0.3, 0.2) for (lo, hi) in BINS]
    hls = [h[[int(np.argmin(abs(TS - t))) for t in TS_GRID]] for h in hls_full]
    # (b) old design on each pool
    ws = np.array([strat(x) for x in wn]); Pw = np.array([(ws == k).mean() for k in range(K)])
    res["old_design"] = []
    for name, (hp, hn) in (("old_pool", old), ("new_pool", new)):
        hs = np.array([strat(x) for x in hn]); Ph = np.array([(hs == k).mean() for k in range(K)])
        for pi in (0.05, 0.2):
            cov = 0
            for r in range(a.R):
                Z = rng.random(4000) < pi; nz = int(Z.sum()); iw = rng.integers(0, len(wp), nz); ih = rng.integers(0, len(hp), 4000 - nz)
                p = np.concatenate([wp[iw], hp[ih]]); n = np.concatenate([wn[iw], hn[ih]]); s = np.array([strat(x) for x in n])
                Ns = np.array([(s == k).sum() for k in range(K)]); pis = pi * Pw / np.maximum(pi * Pw + (1 - pi) * Ph, 1e-300)
                L, U, _ = bounds_corpus(p, n, hls_full, 0.05); tg = float((Ns / 4000 * pis).sum()); cov += L <= tg <= U
            row = {"pool": name, "pi": pi, "coverage": cov / a.R, "CI95": [round(x, 3) for x in cp(cov, a.R)]}; res["old_design"].append(row); print(row, flush=True)
    # (c) new bound on each pool (realized share, without replacement)
    res["new_bound"] = []
    for name, (hp, hn) in (("old_pool", old), ("new_pool", new)):
        for pi in (0.05, 0.2):
            Kw = int(round(pi * 2000)); cov = 0
            for r in range(a.R):
                iw = rng.choice(len(wp), Kw, replace=False); ih = rng.choice(len(hp), 2000 - Kw, replace=False)
                p = np.concatenate([wp[iw], hp[ih]]); n = np.concatenate([wn[iw], hn[ih]]); s = np.array([strat(x) for x in n])
                L, U = realized_bounds(p, s, K, TS_GRID, hls, 0.05); cov += L <= pi <= U
            row = {"pool": name, "pi": pi, "coverage": cov / a.R, "CI95": [round(x, 3) for x in cp(cov, a.R)]}; res["new_bound"].append(row); print(row, flush=True)
    # (d) R2 with independent reference splits
    hp, hn = new; res["R2_splits"] = []
    for split in range(5):
        perm = rng.permutation(len(hp)); ref, aud = perm[:4000], perm[4000:]
        band = null_band_from_reference(hp[ref], np.array([strat(x) for x in hn[ref]]), K, TS_GRID, 0.025)
        for pi in (0.05, 0.2):
            Kw = int(round(pi * 2000)); cov = 0; U = []
            for r in range(100):
                iw = rng.choice(len(wp), Kw, replace=False); ih = rng.choice(aud, 2000 - Kw, replace=False)
                p = np.concatenate([wp[iw], hp[ih]]); n = np.concatenate([wn[iw], hn[ih]]); s = np.array([strat(x) for x in n])
                L2, U2 = realized_bounds(p, s, K, TS_GRID, hls, 0.025, nullband=band); cov += L2 <= pi <= U2; U.append(U2)
            row = {"split": split, "pi": pi, "coverage": cov / 100, "mean_U": round(float(np.mean(U)), 4)}; res["R2_splits"].append(row); print(row, flush=True)
    json.dump(res, open(f"{a.out}/{a.tag}_{a.scheme}_crosscheck.json", "w"), indent=1); print(res["batch_diff"])

if __name__ == "__main__":
    main()
