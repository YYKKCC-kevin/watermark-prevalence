"""Sharp out-of-class counterexample for the Gumbel envelope bound.

DGP: every token has top probability q_const (so max P = q_const), every watermarked doc has eps = eps_const.
Assumed class A(0.3, delta): valid iff q_const <= 1-delta.  With q_const=0.9 and assumed delta=0.2 (max P <= 0.8)
the true power is below the envelope and the DKW upper bound undercovers; assumed delta=0.1 is in-class.
Run: python3 outclass.py --out ../figures [--reps 100]
"""
import argparse, json, os
import numpy as np
from certified import certified_envelope, band_bounds
from dgp import draw_docs_const

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../figures"); ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args(); rng = np.random.default_rng(5)
    ts = np.concatenate([[0.001, 0.005], np.linspace(0.01, 0.99, 99)])
    n, N, pi = 500, 20000, 0.2
    rows = []
    hls = {d: certified_envelope(n, 0.3, d, ts) for d in (0.1, 0.2)}
    for nrest in (1, 9):            # 1: P=(q,1-q) rest concentrated (least signal); 9: rest spread (low-prob tokens give strong signal)
        for q in (0.85, 0.9):
            for d in (0.1, 0.2):
                hl = hls[d]; cov = 0; ucbs = []
                for r in range(a.reps):
                    p, _ = draw_docs_const(N, n, pi, rng, q_const=q, eps_const=0.3, nrest=nrest)
                    lcb, ucb = band_bounds(p, hl, ts); cov += (lcb <= pi <= ucb); ucbs.append(ucb)
                rows.append({"q_const": q, "rest_tokens": nrest, "eps_const": 0.3, "assumed_eps_min": 0.3, "assumed_delta": d,
                             "maxP_in_class": bool(q <= 1 - d + 1e-12), "n": n, "N": N, "pi": pi, "coverage": cov / a.reps,
                             "mean_ucb": round(float(np.mean(ucbs)), 4), "reps": a.reps})
                print(rows[-1], flush=True)
    with open(os.path.join(a.out, "outclass_results.json"), "w") as f:
        json.dump(rows, f, indent=1)

if __name__ == "__main__":
    main()
