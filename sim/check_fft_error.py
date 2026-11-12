"""Floating-point check of the FFT n-fold convolution used by certified.certified_envelope.
Direct convolution of NONNEGATIVE vectors has relative rounding error <= (#terms) * 2^-53 per entry (no cancellation),
so it serves as a reference. We compare the tail probabilities P(sum >= k ds) from both methods.
Run: python3 -I check_fft_error.py"""
import numpy as np, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from envelope import F_bar
def pmf_lattice(eps_min, delta, ds, smax):
    K = int(round(smax / ds)); s = np.arange(K + 1) * ds
    FX = F_bar(1 - np.exp(-s), eps_min, delta); p = np.empty(K + 1); p[:K] = np.diff(FX); p[K] = 1 - FX[K]; return p
out = []
for (n, em, d, ds, smax) in [(10, 0.3, 0.2, 0.01, 30.0), (20, 0.3, 0.1, 0.01, 30.0), (40, 0.3, 0.2, 0.02, 30.0)]:
    p = pmf_lattice(em, d, ds, smax); K = len(p) - 1
    L = 1 << int(np.ceil(np.log2(n * K + 1))); g_fft = np.fft.irfft(np.fft.rfft(p, L) ** n, L)[: n * K + 1]
    t0 = time.time(); g = p.copy()
    for _ in range(n - 1): g = np.convolve(g, p)
    tail_fft = np.cumsum(np.clip(g_fft, 0, None)[::-1])[::-1]; tail_dir = np.cumsum(g[::-1])[::-1]
    err = np.abs(tail_fft - tail_dir).max(); bound_dir = n * (K + 1) * n * K * 2.0 ** -53
    out.append((n, em, d, ds, float(err), float(bound_dir), round(time.time() - t0, 1)))
    print(f"n={n} eps={em} delta={d} ds={ds}: max |tail_FFT - tail_direct| = {err:.2e}; crude bound on direct-sum error {bound_dir:.1e}; tolerance used 1e-8", flush=True)
