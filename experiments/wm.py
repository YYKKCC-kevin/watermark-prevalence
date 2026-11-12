"""Gumbel-max and KGW watermarks: generation and scoring with a context-hash key.

Context: the h previous token ids.  seed(ctx) = hash(key, ctx).  For each step a torch.Generator seeded by seed(ctx)
produces (Gumbel) U ~ U(0,1)^{|V|}  or (KGW) a random permutation whose first gamma*|V| entries form the green list.
Gumbel-max:  w = argmax_w U_w^{1/P(w)}  (distortion-free, temperature 1), pivot Y = U_w.
KGW (soft):  logits[green] += b, sample from softmax, pivot Y = 1{w in green}.
Scoring recomputes the per-position randomness from the context and the observed token; positions whose context
h-gram already occurred earlier in the same document are dropped (dedup), as are the first h positions.
Also recorded at generation time: the unwatermarked NTP max probability of each position (for the (S1) check).
"""
import hashlib, numpy as np, torch
from scipy import stats

def ctx_seed(key: int, ctx) -> int:
    hsh = hashlib.sha256((str(key) + "|" + ",".join(map(str, ctx))).encode()).digest()
    return int.from_bytes(hsh[:8], "little") % (2**63 - 1)

class Watermarker:
    def __init__(self, scheme, key, vocab_size, h=3, gamma=0.5, b=2.0, device="cuda"):
        assert scheme in ("gumbel", "kgw")
        self.scheme, self.key, self.V, self.h, self.gamma, self.b, self.device = scheme, key, vocab_size, h, gamma, b, device
        self.m = int(round(gamma * vocab_size))
        self._gen = torch.Generator(device=device)

    def _rand(self, ctx):
        self._gen.manual_seed(ctx_seed(self.key, ctx))
        if self.scheme == "gumbel":
            return torch.rand(self.V, generator=self._gen, device=self.device)
        return torch.randperm(self.V, generator=self._gen, device=self.device)[: self.m]

    @torch.no_grad()
    def step(self, logits, ctx_batch):
        """logits: (B, V) unwatermarked next-token logits (V may exceed self.V by padding rows); ctx_batch: B contexts.
        Returns (tokens (B,), pivots (B,), maxprob (B,))."""
        logits = logits.float()[:, : self.V]
        probs = torch.softmax(logits, -1)
        maxp = probs.max(-1).values
        toks = torch.empty(len(ctx_batch), dtype=torch.long, device=self.device)
        piv = torch.empty(len(ctx_batch), dtype=torch.float32, device=self.device)
        for i, ctx in enumerate(ctx_batch):
            r = self._rand(ctx)
            if self.scheme == "gumbel":
                p = probs[i].clamp_min(1e-12)
                w = torch.argmax(torch.log(r) / p)          # argmax u^{1/p}
                toks[i] = w; piv[i] = r[w]
            else:
                lg = logits[i].clone(); lg[r] += self.b
                w = torch.multinomial(torch.softmax(lg, -1), 1)[0]
                toks[i] = w; piv[i] = float(bool((r == w).any()))
        return toks, piv, maxp

    @torch.no_grad()
    def score(self, ids, start=0):
        """Pivots for a token sequence, positions t >= max(start, h), with context dedup.
        Returns (pivots np.array, n_scored, n_dropped_dup)."""
        ids = [int(x) for x in ids]
        seen = set(); out = []; dup = 0
        for t in range(max(start, self.h), len(ids)):
            ctx = tuple(ids[t - self.h:t])
            if ctx in seen: dup += 1; continue
            seen.add(ctx)
            r = self._rand(ctx); w = ids[t]
            if w >= self.V: continue
            out.append(float(r[w]) if self.scheme == "gumbel" else float(bool((r == w).any())))
        return np.array(out), len(out), dup

def doc_pvalue(piv, scheme, gamma=0.5, rng=None):
    """Gumbel: p = Gamma_n.sf(sum -log(1-Y)).  KGW: randomized binomial p-value (exactly U(0,1) under H0)."""
    n = len(piv)
    if n == 0: return np.nan
    if scheme == "gumbel":
        y = np.clip(piv, 0.0, 1 - 1e-7)
        return float(stats.gamma.sf((-np.log1p(-y)).sum(), a=n))
    T = int(piv.sum()); u = (rng or np.random.default_rng()).random()
    return float(stats.binom.sf(T, n, gamma) + u * stats.binom.pmf(T, n, gamma))
