"""Sensitivity of the realized-share bounds to null miscalibration (must-do item 3).
Human p-values have CDF H(t) = t - eta * 4 t (1 - t) (eta > 0: fewer small p-values than uniform, threatening the upper bound; eta < 0: more, threatening the lower bound),
so the exact-null upper bound is too low by roughly eta / (h - t).  Watermarked documents follow the least-favourable
mechanism (n = 200, delta = 0.2, eps = 0.3), or ('strong') a 50/50 mixture of such documents and strong ones.
The reference sample is redrawn in every replication, so R2 coverage includes the reference band's own failure probability.  We vary eta and the corpus size N and compare
  exact : Proposition R1 (assumes H(t) = t)
  calib : Proposition R2 with a DKW band from an independent human reference sample of size M (alpha split 0.025/0.025).
Prediction: the exact-null bound fails once eta exceeds the slack ~ sqrt(log(2KJ/alpha)/(2N)); calibration restores it.
Run: python3 -I null_sensitivity.py --out ../figures [--R 200]"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certified import certified_envelope
from realized import realized_bounds, null_band_from_reference
from weak_signal import wm_pvalues, TS_GRID

def human_p(m, eta, rng):
    # inverse CDF of H(t) = t - 4 eta t (1 - t) = (1 - 4 eta) t + 4 eta t^2; eta < 0 gives more small p-values than uniform
    u = rng.random(m)
    if eta == 0: return u
    a, b = 4 * eta, 1 - 4 * eta
    return (-b + np.sqrt(b * b + 4 * a * u)) / (2 * a)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../figures"); ap.add_argument("--R", type=int, default=200)
    a = ap.parse_args(); rng = np.random.default_rng(99); hl = certified_envelope(200, 0.3, 0.2, TS_GRID); rows = []; t0 = time.time()
    for alt in ("LF", "strong"):
     for eta in (0.0, 0.005, 0.01, 0.02, -0.005, -0.01, -0.02):
        for N in (2000, 20000, 100000):
            for pi in (0.05, 0.2):
                for M in (4000, 40000):
                    K = int(round(pi * N)); ce = cc = 0; E = []; C = []
                    for r in range(a.R if N < 100000 else max(50, a.R // 4)):
                        pw = wm_pvalues(K, 200, 0.2, rng, mixed=(alt == 'strong')); ph = human_p(N - K, eta, rng); p = np.concatenate([pw, ph]); s = np.zeros(N, int)
                        Le, Ue = realized_bounds(p, s, 1, TS_GRID, [hl], 0.05)
                        band = null_band_from_reference(human_p(M, eta, rng), np.zeros(M, int), 1, TS_GRID, 0.025)
                        Lc, Uc = realized_bounds(p, s, 1, TS_GRID, [hl], 0.025, nullband=band)
                        ce += Le <= pi <= Ue; cc += Lc <= pi <= Uc; E.append((Le, Ue)); C.append((Lc, Uc))
                    R = len(E); E = np.array(E); C = np.array(C)
                    row = {"alt": alt, "eta": eta, "N": N, "pi": pi, "M_ref": M, "R": R, "exact_cov": ce / R, "exact_mean": E.mean(0).round(4).tolist(),
                           "calib_cov": cc / R, "calib_mean": C.mean(0).round(4).tolist()}
                    rows.append(row); print(json.dumps(row), f"{time.time()-t0:.0f}s", flush=True)
    json.dump(rows, open(os.path.join(a.out, "null_sensitivity_results.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
