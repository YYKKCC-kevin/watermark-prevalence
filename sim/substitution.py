"""Substitution edits at intact contexts (Stage 4, fix for referee R1 W3).

Gumbel-max: if the emitted token w is replaced by w' (chosen from the text only, not from the key), the scored pivot
is U_{w'} conditional on w having won.  Given that w wins, a := U_w^{1/P(w)} ~ U(0,1) and U_v | a ~ U(0, a^{P(v)})
independently, so U_{w'} = a^q V with a, V iid U(0,1), q = P(w').  Its CDF is
    F_sub(x; q) = x^{1/q} + x (1 - x^{1/q - 1}) / (1 - q),   with limit x(1 - log x) as q -> 1,
which is >= x (sub-null).  It is increasing in q, so under max P <= 1 - delta the worst case is q = 1 - delta.
KGW: P(w' green | w emitted, membership of w) >= g / (g + (1 - g)(1 + c q)), g = (gamma|V| - 1)/(|V| - 1), c = e^b - 1.
Run: python3 -I substitution.py
"""
import numpy as np, itertools

def F_sub(x, q):
    x = np.asarray(x, float)
    if q >= 1 - 1e-12: return x * (1 - np.log(np.clip(x, 1e-300, 1)))
    return x ** (1 / q) + x * (1 - x ** (1 / q - 1)) / (1 - q)

def g_sub_kgw(gamma, b, q, V=None):
    # V=None: use gamma - 1e-4, a lower bound on (gamma|V|-1)/(|V|-1) for every |V| >= 10^4 (conservative)
    g = gamma - 1e-4 if V is None else (gamma * V - 1) / (V - 1)
    c = np.exp(b) - 1
    return g / (g + (1 - g) * (1 + c * q))

def main():
    rng = np.random.default_rng(3); xs = np.linspace(0.01, 0.99, 99)
    # 1. monotonicity in q
    qs = np.linspace(0.01, 0.999, 200); Fs = np.array([F_sub(xs, q) for q in qs])
    print("F_sub nondecreasing in q:", bool((np.diff(Fs, axis=0) >= -1e-12).all()), " min F_sub - x:", float((Fs - xs).min()))
    # 2. Monte Carlo check of the Gumbel formula for several NTPs, substitute = a fixed other token
    M = 400000
    for P, j in [(np.array([0.5, 0.3, 0.2]), 1), (np.array([0.2, 0.8]), 1), (np.r_[0.05, 0.6, np.full(7, 0.05)], 1)]:
        U = rng.random((M, len(P))); w = np.argmax(np.log(U) / P, 1); sel = w == 0   # emitted token 0, substitute token j
        Y = U[sel, j]; emp = np.array([(Y <= x).mean() for x in xs]); th = F_sub(xs, P[j])
        print(f"Gumbel P={P[:3].round(2)} q={P[j]}: max|emp-formula|={np.abs(emp-th).max():.4f} (n={sel.sum()})")
    # 3. KGW: exact enumeration over green lists for a small vocabulary, worst case over emitted/substitute pairs
    V, m, b = 10, 5, 2.0; c = np.exp(b) - 1; worst = 1.0
    for trial in range(300):
        P = rng.dirichlet(np.ones(V) * rng.choice([0.2, 1.0]))
        num = np.zeros((V, V)); den = np.zeros(V)
        for G in itertools.combinations(range(V), m):
            g = np.zeros(V, bool); g[list(G)] = True; S = P[g].sum()
            emit = P * np.where(g, np.exp(b), 1.0) / (1 + c * S)
            for w in range(V):
                den[w] += emit[w]; num[w] += emit[w] * g
        cond = num / den[:, None]
        for w in range(V):
            for v in range(V):
                if v == w: continue
                bound = g_sub_kgw(m / V, b, P[v], V)
                worst = min(worst, cond[w, v] - bound)
    print("KGW: min over trials of [P(w' green | w emitted) - bound] =", round(float(worst), 5), "(>= 0 means the bound holds)")
    # 4. size of the effect
    for d in (0.1, 0.2):
        q = 1 - d; s = np.linspace(0, 40, 400001); x = 1 - np.exp(-s)
        mean_score = float(np.trapz(1 - F_sub(x, q), s))
        print(f"delta={d}: worst-case substitute mean score {mean_score:.3f} (null 1); KGW green floor {g_sub_kgw(0.5, 2.0, q):.3f} (null 0.5)")

if __name__ == "__main__":
    main()
