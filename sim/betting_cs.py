"""Anytime-valid confidence sequences for watermark prevalence (fixed-length stratum n).

(A) Fixed-pi hedged betting CS.  Model: Z_i iid Bernoulli(pi) (independent of the past); given Z_i=1 the
    mechanism is arbitrary (adaptive allowed) inside A(eps_min, delta); given Z_i=0 pivots are U(0,1).
    For threshold t and B_i = 1{p_i <= t}:  E[B_i | F_{i-1}] = t + pi (G_i(t) - t)  with  h(t) <= G_i(t) <= 1.
    Under pi >= pi0:   E[B_i|F] >= mu_L(pi0) := t + pi0 (h_lower(t)-t)  -> K_up(pi0) = prod (1 + lam_i (mu_L - B_i))
    Under pi <= pi0:   E[B_i|F] <= mu_U(pi0) := t + pi0 (1-t)           -> K_lo(pi0) = prod (1 + lam_i (B_i - mu_U))
    lam_i predictable in [0, 1/2]  =>  both are nonnegative test supermartingales under pi = pi0, so is the
    hedged mixture (1/2)(K_up + K_lo), and so is its average over thresholds t_1..t_J.  Ville => the set of
    non-rejected pi0 is a (1-alpha)-CS; running intersection keeps validity.  Since the bets do not depend on pi0,
    each K_up factor is positive, affine and increasing in pi0 and each K_lo factor positive, affine and decreasing,
    so K_k(pi0) is convex in pi0 and the non-rejected set is an interval.  On a finite pi0-grid we report the OUTER
    approximation (neighbouring rejected grid points) so an off-grid true pi is never dropped.

(B) Running realized-fraction CS.  Z_i arbitrary (adaptive).  mu_i := E[B_i | F_{i-1}, Z_i] satisfies
    t + Z_i (h_lower(t)-t) <= mu_i <= t + Z_i (1-t).   S_k = sum (B_i - mu_i) is a martingale with increments
    in [-1,1], Hoeffding sub-Gaussian with V_k = k/4.  Robbins normal-mixture boundary
    M_k = sqrt(rho/(V_k+rho)) exp(S_k^2 / (2(V_k+rho)))  (Gaussian mixture over lambda ~ N(0,1/rho)) is a nonneg
    supermartingale; Ville at level alpha gives the two-sided boundary
    bd_k = sqrt(2 (V_k+rho) log( sqrt((V_k+rho)/rho) / alpha ))  with  P(exists k: |S_k| >= bd_k) <= alpha.
    Hence simultaneously for all k:  (sum(B_i-t) - bd_k)/(k(1-t)) <= pibar_k <= (sum(B_i-t) + bd_k)/(k a),
    a = h_lower(t)-t, pibar_k = (1/k) sum_{i<=k} Z_i.

Run: python3 betting_cs.py --out ../figures [--reps 100] [--N 20000]
"""
import argparse, json, os, time
import numpy as np
from scipy.special import logsumexp
from certified import certified_envelope
from dgp import draw_docs

def predictable_lambda(B, alpha, lam_max=0.5):
    """Waudby-Smith & Ramdas predictable plug-in bets (uses only B_1..B_{i-1})."""
    N = len(B); i = np.arange(1, N + 1)
    mu_hat = (0.5 + np.cumsum(B)) / (i + 1)
    sig2 = (0.25 + np.cumsum((B - mu_hat) ** 2)) / (i + 1)
    sig2_prev = np.concatenate([[0.25], sig2[:-1]])
    lam = np.sqrt(2 * np.log(2 / alpha) / (sig2_prev * i * np.log1p(i)))
    return np.minimum(lam, lam_max)

def cs_fixed_pi(p, ts, hl, alpha=0.05, grid=None, block=1000, lam_max=0.5):
    """Return (L, U): running-intersected CS endpoints at every k=1..N."""
    N = len(p); ts = np.asarray(ts, float); hl = np.asarray(hl, float)
    grid = np.linspace(0, 1, 501) if grid is None else grid
    J = len(ts)
    B = (p[:, None] <= ts[None, :]).astype(float)            # N x J
    lam = np.column_stack([predictable_lambda(B[:, j], alpha, lam_max) for j in range(J)])
    muL = ts[None, :] + grid[:, None] * (hl[None, :] - ts[None, :])   # G x J
    muU = ts[None, :] + grid[:, None] * (1 - ts[None, :])
    logK_up = np.zeros((len(grid), J)); logK_lo = np.zeros((len(grid), J))
    thr = np.log(1 / alpha); L = np.empty(N); U = np.empty(N); runL, runU = 0.0, 1.0
    for s in range(0, N, block):
        b = B[s:s + block]; l = lam[s:s + block]
        fu = np.log1p(l[:, None, :] * (muL[None] - b[:, None, :]))
        fl = np.log1p(l[:, None, :] * (b[:, None, :] - muU[None]))
        cu = logK_up[None] + np.cumsum(fu, 0); cl = logK_lo[None] + np.cumsum(fl, 0)
        logK_up, logK_lo = cu[-1], cl[-1]
        logK = logsumexp(np.stack([cu, cl], -1), axis=(2, 3)) - np.log(2 * J)   # blk x G
        ok = logK < thr
        for r in range(len(b)):
            idx = np.flatnonzero(ok[r])
            if len(idx):
                # outer approximation: extend to the neighbouring REJECTED grid points so that an off-grid true pi
                # lying between the last accepted point and the true boundary is not dropped (Codex round 4, item 7).
                lo_i = max(idx[0] - 1, 0); hi_i = min(idx[-1] + 1, len(grid) - 1)
                runL = max(runL, grid[lo_i]); runU = min(runU, grid[hi_i])
            L[s + r] = runL; U[s + r] = runU
    return L, U

def cs_running_fraction(p, t, hl_t, alpha=0.05, rho=None):
    N = len(p); k = np.arange(1, N + 1)
    B = (p <= t).astype(float); S = np.cumsum(B - t)
    V = k / 4.0; rho = (N / 4.0) / 10 if rho is None else rho     # tuned for mid-horizon
    bd = np.sqrt(2 * (V + rho) * np.log(np.sqrt((V + rho) / rho) / alpha))   # two-sided normal mixture, level alpha
    a = hl_t - t
    U = np.minimum(1.0, (S + bd) / (k * a)); L = np.maximum(0.0, (S - bd) / (k * (1 - t)))
    return L, U

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../figures"); ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--N", type=int, default=20000); ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--only", default="all", choices=["all", "running"])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    outpath = os.path.join(a.out, "betting_cs_results.json")
    rng = np.random.default_rng(11)
    ts = np.array([0.01, 0.05, 0.1, 0.2, 0.5]); tgrid = np.concatenate([[0.001, 0.005], np.linspace(0.01, 0.99, 99)])
    checkpoints = [c for c in (200, 500, 1000, 2000, 5000, 10000, 20000, 50000) if c <= a.N]
    res = {"alpha": a.alpha, "ts": ts.tolist(), "N": a.N, "reps": a.reps, "fixed_pi": [], "running": []}
    if a.only == "running" and os.path.exists(outpath):
        res = json.load(open(outpath)); res["running"] = []

    # ---------- (A) fixed-pi CS ----------
    configs = [] if a.only == "running" else [(500, 0.3, 0.2, (0.0, 0.05, 0.2013)), (200, 0.3, 0.2, (0.2013,))]   # 0.2013 is OFF the 0.002 grid on purpose
    for n, em, d, pis in configs:
        hl_full = certified_envelope(n, em, d, tgrid)
        hl = np.array([hl_full[np.argmin(abs(tgrid - t))] for t in ts])
        for pi in pis:
            t0 = time.time(); cov_all = 0; Ls = []; Us = []
            for r in range(a.reps):
                p, Z = draw_docs(a.N, n, pi, rng)
                L, U = cs_fixed_pi(p, ts, hl, a.alpha)
                cov_all += bool(np.all((L <= pi) & (pi <= U)))
                Ls.append(L[np.array(checkpoints) - 1]); Us.append(U[np.array(checkpoints) - 1])
            Ls = np.array(Ls); Us = np.array(Us)
            row = {"n": n, "assumed_eps_min": em, "assumed_delta": d, "pi": pi, "coverage_all_k": cov_all / a.reps,
                   "checkpoints": checkpoints, "mean_L": Ls.mean(0).round(4).tolist(), "mean_U": Us.mean(0).round(4).tolist(),
                   "mean_width": (Us - Ls).mean(0).round(4).tolist(), "sec": round(time.time() - t0, 1)}
            if pi == 0.0:
                row["false_alarm_any_k"] = float(np.mean(Ls[:, -1] > 0))   # L_N>0 means 'some watermark' declared
            res["fixed_pi"].append(row); print(row, flush=True)

    # ---------- (B) running realized fraction with drifting / adversarial Z ----------
    n, em, d = 500, 0.3, 0.2
    hl_full = certified_envelope(n, em, d, tgrid); t = 0.05; hl_t = hl_full[np.argmin(abs(tgrid - t))]
    N = a.N; k = np.arange(1, N + 1)
    patterns = {
        "drift_0_0.4_0.1": np.concatenate([np.zeros(N // 4), np.ones(N // 2) * 0.4, np.ones(N - N // 4 - N // 2) * 0.1]),
        "periodic_bursts": (np.sin(np.arange(N) / 500.0) > 0.6).astype(float),
        "iid_0.2": np.full(N, 0.2),
    }
    for name, prob in patterns.items():
        t0 = time.time(); cov = 0; curves = []
        for r in range(a.reps):
            Z = rng.random(N) < prob
            p, _ = draw_docs(N, n, None, rng, Z=Z)
            L, U = cs_running_fraction(p, t, hl_t, a.alpha)
            pibar = np.cumsum(Z) / k
            cov += bool(np.all((L <= pibar) & (pibar <= U)))
            if r < 3: curves.append({"L": L[np.array(checkpoints) - 1].round(4).tolist(), "U": U[np.array(checkpoints) - 1].round(4).tolist(),
                                     "pibar": pibar[np.array(checkpoints) - 1].round(4).tolist()})
        row = {"pattern": name, "n": n, "t": t, "coverage_all_k": cov / a.reps, "checkpoints": checkpoints, "examples": curves, "sec": round(time.time() - t0, 1)}
        res["running"].append(row); print(row, flush=True)

    with open(outpath, "w") as f:
        json.dump(res, f, indent=1)

if __name__ == "__main__":
    main()
