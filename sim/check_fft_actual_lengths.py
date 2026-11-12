"""Numerical checks of the conservative envelope at the lengths used in the paper (must-do item 4).
(1) Floating point: FFT n-fold convolution vs direct convolution (sums of nonnegative terms, no cancellation) of the same
    lattice pmf, comparing tail probabilities P(sum >= k ds), at n = 50, 100, 200 (step 0.02) and 300 (step 0.05).
(2) Discretisation: envelope at step 0.01 (used in the paper) vs 0.005 and 0.0025, to show how conservative the lattice is.
Run: python3 -I check_fft_actual_lengths.py --out ../figures"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envelope import F_bar
from certified import certified_envelope

def pmf_lattice(eps_min, delta, ds, smax):
    K = int(round(smax / ds)); s = np.arange(K + 1) * ds
    FX = F_bar(1 - np.exp(-s), eps_min, delta); p = np.empty(K + 1); p[:K] = np.diff(FX); p[K] = 1 - FX[K]; return p

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../figures"); a = ap.parse_args(); res = {"fft_vs_direct": [], "lattice": []}
    for (n, ds, smax) in [(50, 0.02, 30.0), (100, 0.02, 30.0), (200, 0.02, 30.0), (300, 0.05, 30.0)]:
        t0 = time.time(); p = pmf_lattice(0.3, 0.2, ds, smax); K = len(p) - 1
        L = 1 << int(np.ceil(np.log2(n * K + 1))); g_fft = np.clip(np.fft.irfft(np.fft.rfft(p, L) ** n, L)[: n * K + 1], 0, None)
        g = p.copy()
        for _ in range(n - 1): g = np.convolve(g, p)
        tf = np.cumsum(g_fft[::-1])[::-1]; td = np.cumsum(g[::-1])[::-1]
        r = {"n": n, "ds": ds, "max_abs_tail_diff": float(np.abs(tf - td).max()), "sec": round(time.time() - t0, 1)}
        res["fft_vs_direct"].append(r); print(r, flush=True)
    ts = np.array([0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9])
    for n in (100, 200, 300):
        h1 = certified_envelope(n, 0.3, 0.2, ts, ds=0.01); h2 = certified_envelope(n, 0.3, 0.2, ts, ds=0.005); h3 = certified_envelope(n, 0.3, 0.2, ts, ds=0.0025)
        r = {"n": n, "ts": ts.tolist(), "h_ds0.01": h1.round(4).tolist(), "h_ds0.005": h2.round(4).tolist(), "h_ds0.0025": h3.round(4).tolist(),
             "max_gain_0.01_to_0.0025": round(float((h3 - h1).max()), 4), "monotone_in_ds": bool((h3 >= h2 - 1e-9).all() and (h2 >= h1 - 1e-9).all())}
        res["lattice"].append(r); print(r, flush=True)
    json.dump(res, open(os.path.join(a.out, "fft_actual_lengths.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
