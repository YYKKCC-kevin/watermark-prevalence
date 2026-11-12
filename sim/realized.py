"""Fixed-N bounds on the REALIZED number/fraction of watermarked documents in an eligible corpus (must-do item 1),
with an optional calibrated null (must-do item 3).

Setting.  N documents with observed lengths and stratum s(i); labels Z_i arbitrary but fixed before document i is seen.
Conditionally on all labels, all lengths and past p-values:
  Z_i = 0  ->  P(p_i <= t) = H_s(t)   (exact null: H_s(t) = t;  calibrated: H_L <= H_s <= H_U from a reference band)
  Z_i = 1  ->  P(p_i <= t) >= h_s(t)  (power envelope for the stratum)
For B_ij = 1{p_i <= t_j}, the increments B_ij - E[B_ij | past] lie in an interval of length 1, so by Azuma-Hoeffding and a
union bound over strata s and thresholds j, with beta_s = sqrt(N_s log(2 K J / alpha) / 2), simultaneously:
  |sum_{i in s} B_ij - sum_{i in s} m_ij| <= beta_s.
Since sum_i m_ij = (N_s - K_s) H_s(t_j) + sum_{i: Z=1} G_i(t_j) with h_s <= G_i <= 1:
  K_s <= (S_sj + beta_s - N_s H) / (h_s(t_j) - H)   for every H in [H_L, H_U] with h > H   (take the max over the endpoints)
  K_s >= (S_sj - beta_s - N_s H) / (1 - H)            (minimised at H = H_U)
Take the best threshold per stratum, clip to [0, N_s], sum over strata.
"""
import numpy as np

def stratum_counts_bounds(B_sum, Ns, ts, hl, beta, HL=None, HU=None):
    """B_sum: (J,) counts of p <= t_j in the stratum; hl: (J,) envelope; HL/HU: (J,) null band (None = exact null)."""
    ts = np.asarray(ts, float); hl = np.asarray(hl, float)
    if HL is None: HL = ts; HU = ts
    HL = np.asarray(HL, float); HU = np.asarray(HU, float)
    lo = (B_sum - beta - Ns * HU) / np.maximum(1 - HU, 1e-12)
    ok = hl > HU + 1e-9
    up = np.full(len(ts), np.inf)
    for H in (HL, HU):
        val = (B_sum + beta - Ns * H) / np.where(ok, hl - H, 1.0)
        up = np.where(ok, np.maximum(np.where(np.isfinite(up), up, -np.inf), val), np.inf)
    K_lo = max(0.0, float(np.max(lo))); K_up = min(float(Ns), float(np.min(up))) if ok.any() else float(Ns)
    return K_lo, max(K_up, 0.0)

def realized_bounds(p, strata, K, ts, hls, alpha=0.05, nullband=None):
    """p: (N,) p-values; strata: (N,) stratum index 0..K-1; hls: list of (J,) envelopes per stratum;
    nullband: None or list of (HL, HU) per stratum.  Returns (L, U) for the realized fraction sum Z / N."""
    p = np.asarray(p); strata = np.asarray(strata); N = len(p); ts = np.asarray(ts); J = len(ts)
    KL = KU = 0.0
    for s in range(K):
        m = strata == s; Ns = int(m.sum())
        if Ns == 0: continue
        beta = np.sqrt(Ns * np.log(2 * K * J / alpha) / 2)
        B_sum = np.array([(p[m] <= t).sum() for t in ts], float)
        HL, HU = (None, None) if nullband is None else nullband[s]
        lo, up = stratum_counts_bounds(B_sum, Ns, ts, hls[s], beta, HL, HU)
        KL += lo; KU += up
    return KL / N, KU / N

def null_band_from_reference(p_ref, strata_ref, K, ts, alpha_ref=0.05):
    """Simultaneous (over strata and t) DKW band for the null CDF of each stratum from an independent human reference
    sample.  Returns list of (HL, HU) arrays; strata without reference documents get the vacuous band [0, 1]."""
    out = []
    for s in range(K):
        m = strata_ref == s; M = int(m.sum())
        if M == 0: out.append((np.zeros(len(ts)), np.ones(len(ts)))); continue
        b = np.sqrt(np.log(2 * K / alpha_ref) / (2 * M)); F = np.array([(p_ref[m] <= t).mean() for t in ts])
        out.append((np.clip(F - b, 0, 1), np.clip(F + b, 0, 1)))
    return out

def envelope_chosen_threshold(ts, hl, Ns_guess=None):
    """Data-free single threshold: minimise sqrt(t(1-t)) / (h(t) - t), the leading term of the upper-end width when
    the prevalence is small (pre-specified from the envelope alone)."""
    ts = np.asarray(ts); hl = np.asarray(hl); ok = hl > ts + 1e-9
    score = np.where(ok, np.sqrt(ts * (1 - ts)) / np.where(ok, hl - ts, 1), np.inf)
    return int(np.argmin(score))
