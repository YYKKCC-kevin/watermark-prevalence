"""Figures from data/results/*_analysis.json and *_sequential.npz.  Usage: python3 plots.py --out data/results"""
import argparse, glob, json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default="data/results"); ap.add_argument("--gen", default="data/gen"); a = ap.parse_args()
    files = sorted(glob.glob(f"{a.res}/*_analysis.json")); R = [json.load(open(f)) for f in files]
    # Fig 1: interval vs truth for each (model, scheme, pool, delta) at edit_eps_lo=0.5 (in-class) and 0.3
    for lo in (0.3, 0.5):
        rows = [r for r in R if abs(r["edit_eps_lo"] - lo) < 1e-9]
        if not rows: continue
        fig, axes = plt.subplots(1, len(rows), figsize=(4 * len(rows), 3.6), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, r in zip(axes, rows):
            x = 0
            for pool in ("scattered", "span", "paraphrase"):
                for d, c in ((0.1, "C0"), (0.2, "C1")):
                    for m in [m for m in r["mixtures"] if m["pool"] == pool]:
                        L, U = m[f"delta{d}"]["mean_L"], m[f"delta{d}"]["mean_U"]
                        ax.plot([x, x], [L, U], color=c, lw=3, alpha=.8); ax.plot(x, m["pi"], "k_", ms=10)
                        ax.plot(x, m["naive_frac_p_le_0.05_mean"], "rx", ms=5); ax.plot(x, m["reference_noedit_point_mean"], "g+", ms=7)
                        x += 1
                x += 1
            ax.set_title(f"{r['tag']} / {r['scheme']}", fontsize=10); ax.set_ylim(-0.02, 1.02); ax.set_xticks([])
        axes[0].set_ylabel("prevalence"); fig.suptitle(f"95% bounds (blue δ=0.1, orange δ=0.2); black = truth, red x = naive frac(p≤.05), green + = no-edit reference estimate; edits ε_lo={lo}", fontsize=9)
        fig.tight_layout(); fig.savefig(f"{a.res}/fig_bounds_editlo{lo}.png", dpi=130); plt.close(fig)
    # Fig 2: sequential CS
    seqs = sorted(glob.glob(f"{a.res}/*_editlo0.5_sequential.npz"))
    if seqs:
        fig, axes = plt.subplots(1, len(seqs), figsize=(4 * len(seqs), 3.2), sharey=True); axes = np.atleast_1d(axes)
        for ax, f in zip(axes, seqs):
            d = np.load(f); k = np.arange(1, len(d["L"]) + 1)
            ax.fill_between(k, d["L"], d["U"], color="C0", alpha=.3, label="running-fraction CS"); ax.plot(k, d["pibar"], "k", lw=1, label="realized fraction")
            ax.set_title(os.path.basename(f).split("_editlo")[0], fontsize=9); ax.set_xlabel("documents"); ax.set_ylim(0, 1)
        axes[0].set_ylabel("prevalence"); axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{a.res}/fig_sequential.png", dpi=130); plt.close(fig)
    # Fig 3: (S1) max-prob histograms
    gens = sorted(glob.glob(f"{a.gen}/*_gumbel.npz"))
    if gens:
        fig, axes = plt.subplots(1, len(gens), figsize=(4 * len(gens), 3)); axes = np.atleast_1d(axes)
        for ax, f in zip(axes, gens):
            d = np.load(f, allow_pickle=True); mp = np.concatenate([np.asarray(x, float) for x in d["maxp"]])
            ax.hist(mp, bins=50, color="gray"); ax.set_title(f"{os.path.basename(f).split('_')[0]}: max_w P_t(w)", fontsize=9); ax.set_yscale("log")
            for q in (0.8, 0.9): ax.axvline(q, color="C3", ls="--", lw=1)
        fig.tight_layout(); fig.savefig(f"{a.res}/fig_maxprob.png", dpi=130); plt.close(fig)
    print("figures written to", a.res)

if __name__ == "__main__":
    main()
