"""v2 main real-model analysis (realized share).  For one (model, scheme) under the original key:
fresh human pool (12000 C4 documents, eligible docs 6600-18599), split once per replication-block into a 4000-document
reference set (for the calibrated null, Proposition R2) and an 8000-document audit set; corpora of N = 2000 eligible
documents with exactly round(pi N) watermarked ones, drawn without replacement (with replacement only when the
paraphrase pool is smaller than needed), human documents without replacement from the audit set.
Pools: scattered and span random edits (eps ~ U[0.3,1]), plausible substitutions, near-floor (eps ~ U[0.3,0.4]),
short (50-100 tokens), LLM paraphrase.  Bounds: Proposition R1 (exact null) over 12 thresholds, R1 at the single
threshold 0.05, and Proposition R2 (calibrated null, alpha 0.025 + 0.025).  delta in {0.1, 0.2}; sigma_max in {0, 0.05}.
Usage: PYTHONPATH=.:../sim python3 realized_perturb.py --tag Qwen3-1p7B --scheme kgw"""
import argparse, json, numpy as np
from analysis import envelope, BINS, strat, TS
from realized import realized_bounds, null_band_from_reference
from calib_analysis import load, TS_GRID, cp

POOLS = {"random edits (scattered)": "_edit_scattered_eps0.3", "random edits (span)": "_edit_span_eps0.3", "substitutions": "_subst",
         "near-floor": "_edit_scattered_eps0.3-0.4", "short": "_edit_scattered_eps0.3_short100", "paraphrase": "_paraphrase"}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--scheme", required=True)
    ap.add_argument("--R", type=int, default=200); ap.add_argument("--N", type=int, default=2000); ap.add_argument("--out", default="data/results")
    a = ap.parse_args(); rng = np.random.default_rng(4321); K = len(BINS)
    hp, hn = load(f"data/calib/k20261111/{a.tag}_{a.scheme}_human.npz"); hs_all = np.array([strat(x) for x in hn])
    idx = [int(np.argmin(abs(TS - t))) for t in TS_GRID]
    env = {(d, sg): [envelope(a.scheme, lo, 0.3, d, sigma=sg)[idx] for (lo, hi) in BINS] for d in (0.1, 0.2) for sg in (0.0, 0.05)}
    res = {"tag": a.tag, "scheme": a.scheme, "N": a.N, "R": a.R, "n_human": int(len(hp)), "rows": []}
    for pname, suf in POOLS.items():
        wp, wn = load(f"data/gen/{a.tag}_{a.scheme}{suf}.npz"); res.setdefault("pool_sizes", {})[pname] = int(len(wp))
        for (d, sg), hls in env.items():
            if sg > 0 and pname not in ("substitutions",): continue
            for pi in (0.05, 0.2):
                Kw = int(round(pi * a.N)); cov = {"multi": 0, "t005": 0, "calib": 0}; B = {k: [] for k in cov}
                for r in range(a.R):
                    if r % 50 == 0:   # new reference/audit split every 50 corpora (4 independent splits per cell)
                        perm = rng.permutation(len(hp)); ref, aud = perm[:4000], perm[4000:]
                        band = null_band_from_reference(hp[ref], hs_all[ref], K, TS_GRID, 0.025)
                    iw = rng.choice(len(wp), Kw, replace=Kw > len(wp)); ih = rng.choice(aud, a.N - Kw, replace=False)
                    p = np.concatenate([wp[iw], hp[ih]]); n = np.concatenate([wn[iw], hn[ih]]); s = np.array([strat(x) for x in n])
                    out = {"multi": realized_bounds(p, s, K, TS_GRID, hls, 0.05),
                           "t005": realized_bounds(p, s, K, TS_GRID[[2]], [h[[2]] for h in hls], 0.05),
                           "calib": realized_bounds(p, s, K, TS_GRID, hls, 0.025, nullband=band)}
                    for k, (L, U) in out.items(): cov[k] += L <= pi <= U; B[k].append((L, U))
                row = {"pool": pname, "delta": d, "sigma_max": sg, "pi": pi, "with_replacement": bool(Kw > len(wp))}
                for k in cov:
                    v = np.array(B[k]); row[k] = {"coverage": cov[k] / a.R, "CI95": [round(x, 3) for x in cp(cov[k], a.R)], "mean": v.mean(0).round(4).tolist()}
                res["rows"].append(row); print(json.dumps(row), flush=True)
    json.dump(res, open(f"{a.out}/{a.tag}_{a.scheme}_realized.json", "w"), indent=1)

if __name__ == "__main__":
    main()
