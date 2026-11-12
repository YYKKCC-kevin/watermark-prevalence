"""Calibration study (must-do item 3).  For one (model, scheme) and three keys:
  - human: 12000 fresh C4 documents scored under each key (data/calib/k<key>/<tag>_<scheme>_human.npz), same texts for all keys;
  - watermarked: scattered-edit pool generated under the SAME key (key 20261111: the original 2000-document pool;
    other keys: 1000-document pools in data/calib/k<key>/);
  - per key: null diagnostics (KS, deviation of the empirical CDF from uniform), and the spread of F_hat(t) across keys on
    the same texts versus its binomial standard error (key variation vs text-sampling variation);
  - held-out coverage of the realized fraction (fixed-N, Proposition R1 exact null vs R2 calibrated null with an
    independent 4000-document human reference split), N = 2000, pi in {0.05, 0.2}, delta in {0.1, 0.2}, 200 corpora,
    human documents drawn without replacement from the 8000-document audit split, watermarked documents without replacement.
Usage: PYTHONPATH=.:../sim python3 calib_analysis.py --tag Qwen3-1p7B --scheme kgw
"""
import argparse, json, os, numpy as np
from scipy import stats
from analysis import envelope, BINS, strat
from realized import realized_bounds, null_band_from_reference

KEYS = [20261111, 20261112, 20261113]
TS_GRID = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

def load(path, nmin=50):
    e = np.load(path, allow_pickle=True); n = e["n"].astype(int) if "n" in e.files else np.array([len(x) for x in e["pivots"]])
    p = e["pvals"].astype(float); m = n >= nmin; return p[m], n[m]

def cp(k, n):
    return stats.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0, stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--scheme", required=True)
    ap.add_argument("--R", type=int, default=200); ap.add_argument("--N", type=int, default=2000); ap.add_argument("--out", default="data/results")
    a = ap.parse_args(); rng = np.random.default_rng(777); K = len(BINS)
    res = {"tag": a.tag, "scheme": a.scheme, "keys": KEYS, "per_key": {}, "coverage": []}
    hum = {k: load(f"data/calib/k{k}/{a.tag}_{a.scheme}_human.npz") for k in KEYS}
    # key variation on the same texts: all keys score the same documents in the same order (same skip / filters)
    n0 = hum[KEYS[0]][1]; same_len = all(len(hum[k][1]) == len(n0) and np.array_equal(hum[k][1], n0) for k in KEYS)
    res["same_texts_same_lengths"] = bool(same_len)
    Fk = {k: np.array([(hum[k][0] <= t).mean() for t in TS_GRID]) for k in KEYS}
    M = len(hum[KEYS[0]][0]); se = np.sqrt(TS_GRID * (1 - TS_GRID) / M)
    res["Fhat_by_key"] = {str(k): Fk[k].round(4).tolist() for k in KEYS}; res["binomial_se"] = se.round(4).tolist()
    res["max_abs_dev_from_uniform_in_se"] = {str(k): round(float(np.max(np.abs(Fk[k] - TS_GRID) / se)), 2) for k in KEYS}
    res["ks"] = {str(k): [round(float(x), 4) for x in stats.kstest(hum[k][0], "uniform")] for k in KEYS}
    wmfile = {KEYS[0]: f"data/gen/{a.tag}_{a.scheme}_edit_scattered_eps0.3.npz"}
    for k in KEYS[1:]: wmfile[k] = f"data/calib/k{k}/{a.tag}_{a.scheme}_edit_scattered_eps0.3.npz"
    for d in (0.1, 0.2):
        hls = [envelope(a.scheme, lo, 0.3, d)[[np.argmin(abs(np.r_[[0.001, 0.005], np.linspace(0.01, 0.99, 99)] - t)) for t in TS_GRID]] for (lo, hi) in BINS]
        for k in KEYS:
            hp, hn = hum[k]; wp, wn = load(wmfile[k])
            perm = rng.permutation(len(hp)); ref, aud = perm[:4000], perm[4000:]
            band = null_band_from_reference(hp[ref], np.array([strat(x) for x in hn[ref]]), K, TS_GRID, alpha_ref=0.025)
            for pi in (0.05, 0.2):
                Kw = int(round(pi * a.N)); covE = covC = 0; LE = []; UE = []; LC = []; UC = []
                for r in range(a.R):
                    iw = rng.choice(len(wp), Kw, replace=False); ih = rng.choice(aud, a.N - Kw, replace=False)
                    p = np.concatenate([wp[iw], hp[ih]]); n = np.concatenate([wn[iw], hn[ih]]); s = np.array([strat(x) for x in n])
                    L1, U1 = realized_bounds(p, s, K, TS_GRID, hls, alpha=0.05)
                    L2, U2 = realized_bounds(p, s, K, TS_GRID, hls, alpha=0.025, nullband=band)
                    covE += L1 <= pi <= U1; covC += L2 <= pi <= U2; LE.append(L1); UE.append(U1); LC.append(L2); UC.append(U2)
                row = {"key": k, "delta": d, "pi": pi, "N": a.N, "R": a.R, "n_wm_pool": int(len(wp)),
                       "exact_null": {"coverage": covE / a.R, "CI95": [round(x, 3) for x in cp(covE, a.R)], "mean": [round(float(np.mean(LE)), 4), round(float(np.mean(UE)), 4)]},
                       "calibrated_null": {"coverage": covC / a.R, "CI95": [round(x, 3) for x in cp(covC, a.R)], "mean": [round(float(np.mean(LC)), 4), round(float(np.mean(UC)), 4)]}}
                res["coverage"].append(row); print(json.dumps(row), flush=True)
    json.dump(res, open(f"{a.out}/{a.tag}_{a.scheme}_calib.json", "w"), indent=1)
    print({k: res[k] for k in ("same_texts_same_lengths", "max_abs_dev_from_uniform_in_se", "ks")})

if __name__ == "__main__":
    main()
