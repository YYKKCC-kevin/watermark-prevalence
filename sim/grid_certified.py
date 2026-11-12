"""Identified-set widths on a FIXED in-class DGP with the CERTIFIED envelope (replaces the Day-1 pi_U_grid figure,
which used the FFT approximation and a delta-dependent DGP).
DGP (same for every cell): eps_i ~ U[0.5, 1], per-token top prob q ~ U[0.3, 0.8] with remainder spread over 9 tokens,
so it lies in A(eps_min, delta) for every eps_min <= 0.5 and delta <= 0.2.  pi = 0.2, N = 20000, R reps, 95% DKW band.
Run: python3 -I grid_certified.py --out ../figures [--reps 20]
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from certified import certified_envelope, band_bounds
from dgp import draw_docs

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../figures"); ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args(); rng = np.random.default_rng(11)
    ts = np.concatenate([np.array([0.001, 0.005]), np.linspace(0.01, 0.99, 99)])
    ns, ems, ds = (100, 200, 500, 1000), (0.2, 0.3, 0.5), (0.05, 0.1, 0.2)
    N, pi = 20000, 0.2; rows = []; t0 = time.time()
    for n in ns:
        hls = {(em, d): certified_envelope(n, em, d, ts) for em in ems for d in ds}
        res = {k: [] for k in hls}
        for r in range(a.reps):
            p, _ = draw_docs(N, n, pi, rng, eps_lo=0.5, q_lo=0.3, q_hi=0.8)
            for k, hl in hls.items(): res[k].append(band_bounds(p, hl, ts))
        for (em, d), v in res.items():
            v = np.array(v); cov = float(np.mean((v[:, 0] <= pi) & (pi <= v[:, 1])))
            rows.append({"n": n, "eps_min": em, "delta": d, "mean_lcb": round(float(v[:, 0].mean()), 4), "mean_ucb": round(float(v[:, 1].mean()), 4), "coverage": cov, "reps": a.reps})
            print(rows[-1], f"{time.time()-t0:.0f}s", flush=True)
    json.dump({"dgp": "eps~U[0.5,1], q~U[0.3,0.8], nrest=9, pi=0.2, N=20000", "rows": rows}, open(os.path.join(a.out, "grid_certified_results.json"), "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
    for ax, d in zip(axs, ds):
        for em, c in zip(ems, ("#1b6ca8", "#e08214", "#2d9d5b")):
            y = [next(r["mean_ucb"] for r in rows if r["n"] == n and r["eps_min"] == em and r["delta"] == d) for n in ns]
            ax.plot(ns, y, "o-", color=c, label=rf"$\varepsilon_{{\min}}={em}$")
        ax.axhline(pi, ls="--", c="k", lw=0.8); ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
        ax.set_title(rf"$\delta={d}$ ($\max P\leq{1-d:.2f}$)", fontsize=10); ax.set_xlabel("document length $n$")
    axs[0].set_ylabel("mean 95% upper bound on $\\pi$"); axs[0].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "grid_certified.pdf")); fig.savefig(os.path.join(a.out, "grid_certified.png"), dpi=150)

if __name__ == "__main__":
    main()
