"""KGW (green-list) watermark: least-favourable green probability, power envelope, degenerate counterexample.

Soft KGW with logit bias b: given NTP P and green set G (uniform gamma-subset of V),
   P(green | P, G) = phi(S) = e^b S / (1 + (e^b-1) S),   S = sum_{w in G} P(w),   E_G S = gamma.
Excess over the null rate:  phi(S) - S = psi(S) := c S(1-S)/(1+cS), c = e^b-1, strictly concave in S (psi'' < 0),
psi(0)=psi(1)=0.  S is linear in P, so P -> E_G psi(S(P)) is concave; its minimum over the polytope
{P>=0, sum P=1, max P <= 1-delta} is attained at a vertex, i.e. the SAME vertex structure as Gumbel Lemma 1:
P* = (1-delta, delta, 0, ...) for delta <= 1/2.  Hence
   mu(P) - gamma >= Delta_KGW(delta) := E_G psi(S(P*)) = 2 p_{10} [psi(delta) + psi(1-delta)] / 2 ... (exact hypergeometric below)
Degenerate NTP (delta=0, P a point mass): S in {0,1} a.s., psi(S)=0, Delta_KGW=0: watermarked pivots are exactly
Bernoulli(gamma) = null law => non-identification (Li et al.'s KGW counterexample in this framework).

Document statistic T_n = sum Y_t ~ Bin(n, gamma) under H0; randomized p-value p = sf0(T) + U f0(T) is exactly U(0,1).
Under Model * in the class, the conditional green prob given history is >= gamma + eps_min Delta_KGW, so by the
stepwise coupling T_n >=_st Bin(n, gamma + eps_min Delta_KGW) and the power envelope is
   h_KGW(t) = sum_k P(Bin(n, mu_bar)=k) clip((t - sf0(k)) / f0(k), 0, 1),   exact (no FFT).

Run: python3 kgw.py --out ../figures
"""
import argparse, json, os, itertools
import numpy as np
from scipy import stats
from envelope import F_star

def psi(S, b):
    c = np.exp(b) - 1.0
    return c * S * (1 - S) / (1 + c * S)

def delta_kgw(gamma, delta, b, V=50000):
    """Exact E_G psi(S(P*)) with G a uniform gamma-subset of a vocabulary of size V (hypergeometric)."""
    if V is None: p10 = gamma * (1 - gamma)           # |V| -> infinity limit; a lower bound on p10 for every |V|
    else:
        m = int(round(gamma * V))
        p10 = m * (V - m) / (V * (V - 1))            # token1 green, token2 not (and symmetric)
    return p10 * (psi(1 - delta, b) + psi(delta, b))   # psi(0)=psi(1)=0 terms vanish

def exact_excess(P, gamma, b, V):
    """E_G psi(S(P)) by enumeration over all gamma-subsets (small V only)."""
    m = int(round(gamma * V)); P = np.asarray(P, float)
    tot = 0.0; cnt = 0
    for G in itertools.combinations(range(V), m):
        tot += psi(P[list(G)].sum(), b); cnt += 1
    return tot / cnt

def h_kgw(ts, n, gamma, eps_min, delta, b, V=50000, sigma=0.0):
    """sigma: share of substituted positions at intact contexts; their green probability is >= g_sub (substitution.py)."""
    mu = gamma + eps_min * delta_kgw(gamma, delta, b, V)
    if sigma > 0:
        from substitution import g_sub_kgw
        mu -= sigma * (gamma - g_sub_kgw(gamma, b, 1 - delta, V))
    k = np.arange(n + 1)
    f0 = stats.binom.pmf(k, n, gamma); sf0 = stats.binom.sf(k, n, gamma)
    pm = stats.binom.pmf(k, n, mu)
    out = []
    for t in ts:
        out.append(float((pm * np.clip((t - sf0) / f0, 0, 1)).sum()))
    return np.array(out), mu

def gumbel_mean_shift(eps_min, delta, ds=1e-4, smax=60.0):
    """E[-log(1-Ybar)] - 1 under the Gumbel least-favourable law (per-token mean shift of the sum statistic)."""
    s = np.arange(0, smax, ds); x = 1 - np.exp(-s)
    Fb = (1 - eps_min) * x + eps_min * F_star(x, delta)
    return float(np.trapz(1 - Fb, s)) - 1.0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="../figures"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(3)
    gamma, b = 0.5, 2.0
    res = {"gamma": gamma, "b": b}

    # 1. vertex minimum check by enumeration (V=12, m=6 -> 924 subsets), random P with max <= 1-delta
    V = 12; checks = []
    for delta in (0.1, 0.2, 0.3, 0.5):
        c = 1 - delta; lf = exact_excess(np.r_[c, 1 - c, np.zeros(V - 2)], gamma, b, V)
        worst = np.inf
        for _ in range(300):
            P = rng.dirichlet(np.ones(V) * rng.choice([0.05, 0.2, 1.0, 5.0]))
            # water-filling onto {max P <= c}: move excess of capped coordinates to the uncapped ones
            for _ in range(200):
                over = P > c
                if not over.any(): break
                excess = (P[over] - c).sum(); P[over] = c
                free = ~over; P[free] += excess * (P[free] / P[free].sum() if P[free].sum() > 0 else 1.0 / free.sum())
            assert P.max() <= c + 1e-9 and abs(P.sum() - 1) < 1e-9
            worst = min(worst, exact_excess(P, gamma, b, V))
        checks.append({"delta": delta, "LF_vertex_excess": round(lf, 5), "min_random_excess": round(worst, 5),
                       "closed_form_hypergeom": round(delta_kgw(gamma, delta, b, V), 5), "vertex_is_min": bool(lf <= worst + 1e-12)})
        print(checks[-1], flush=True)
    res["vertex_min_check"] = checks

    # 2. null uniformity of randomized p-value and MC check of h_kgw under the LF mechanism
    n = 200; eps_min, delta = 0.3, 0.2
    ts = np.array([0.01, 0.05, 0.1, 0.5]); h, mu = h_kgw(ts, n, gamma, eps_min, delta, b)
    M = 200000
    T0 = rng.binomial(n, gamma, M); p0 = stats.binom.sf(T0, n, gamma) + rng.random(M) * stats.binom.pmf(T0, n, gamma)
    T1 = rng.binomial(n, mu, M); p1 = stats.binom.sf(T1, n, gamma) + rng.random(M) * stats.binom.pmf(T1, n, gamma)
    res["mc_check"] = {"n": n, "eps_min": eps_min, "delta": delta, "mu_bar": round(mu, 5), "ts": ts.tolist(),
                       "null_cdf": [round(float((p0 <= t).mean()), 4) for t in ts], "h_kgw": h.round(4).tolist(),
                       "h_mc_LF": [round(float((p1 <= t).mean()), 4) for t in ts]}
    print(res["mc_check"], flush=True)

    # 3. per-token signal vs delta (both schemes) and degenerate limit
    rows = []
    for delta in (0.0, 0.01, 0.05, 0.1, 0.2, 0.5):
        rows.append({"delta": delta, "Delta_KGW_excess_green_prob": round(delta_kgw(gamma, delta, b), 5) if delta > 0 else 0.0,
                     "Gumbel_mean_shift_eps1": round(gumbel_mean_shift(1.0, delta), 5) if delta > 0 else 0.0})
    res["signal_vs_delta"] = rows; print(rows, flush=True)

    # 4. envelopes at matched (n, eps_min, delta): h(0.05)
    from certified import certified_envelope
    tt = np.array([0.05]); comp = []
    for n in (200, 500, 1000):
        for delta in (0.1, 0.2):
            hk, _ = h_kgw(tt, n, gamma, 0.3, delta, b); hg = certified_envelope(n, 0.3, delta, tt)
            comp.append({"n": n, "eps_min": 0.3, "delta": delta, "h_KGW(0.05)": round(float(hk[0]), 4), "h_Gumbel_lower(0.05)": round(float(hg[0]), 4)})
    hk0, _ = h_kgw(tt, 500, gamma, 0.3, 0.0, b)
    res["envelope_comparison"] = comp; res["degenerate_delta0_h_kgw_0.05"] = round(float(hk0[0]), 6)
    print(comp, "degenerate:", res["degenerate_delta0_h_kgw_0.05"], flush=True)
    with open(os.path.join(a.out, "kgw_results.json"), "w") as f:
        json.dump(res, f, indent=1)

if __name__ == "__main__":
    main()
