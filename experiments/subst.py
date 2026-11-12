"""Plausible-substitution edits (Stage 4; referee R1 W3).  A substituted token is drawn from the generating model's own
next-token distribution at that position, with the emitted token excluded.  The choice uses only the text, never the key,
which is the condition of the substitution bound in sim/substitution.py.  Substitutions are spaced at least h+1 apart, so
each substituted token keeps its intact h-gram context and its pivot is the key value of a token that lost the argmax
(sub-null).  The h positions after a substitution get a new context (null pivot).
Per document: substitution rate r_i ~ U[r_lo, r_hi]; realized shares over scored positions:
  eps_eff = token and context original;  sig_eff = substituted token with original context;  rest = null.
Usage: CUDA_VISIBLE_DEVICES=2 python3 subst.py --npz data/gen/opt-1p3b_gumbel.npz
"""
import argparse, numpy as np, torch
from transformers import AutoModelForCausalLM
from wm import Watermarker, doc_pvalue

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--r_lo", type=float, default=0.05)
    ap.add_argument("--r_hi", type=float, default=0.35); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args(); torch.set_grad_enabled(False)
    d = np.load(a.npz, allow_pickle=True); scheme = str(d["scheme"]); key = int(d["key"]); h = int(d["h"]); V = int(d["V"]); model_name = str(d["model"])
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).cuda().eval()
    wm = Watermarker(scheme, key, V, h=h); rng = np.random.default_rng(a.seed); gen = torch.Generator(device="cuda").manual_seed(a.seed)
    ids_all, plen_all = d["ids"], d["prompt_len"]; N = len(ids_all)
    piv_out, pv_out, ns, eps_eff, sig_eff, rates = [], [], [], [], [], []
    for s in range(0, N, a.batch):
        chunk = [np.asarray(x, np.int64) for x in ids_all[s:s + a.batch]]; L = max(len(x) for x in chunk)
        inp = torch.zeros((len(chunk), L), dtype=torch.long); att = torch.zeros((len(chunk), L), dtype=torch.long)
        for j, x in enumerate(chunk): inp[j, :len(x)] = torch.from_numpy(x); att[j, :len(x)] = 1
        logits = model(input_ids=inp.cuda(), attention_mask=att.cuda()).logits
        for j, x in enumerate(chunk):
            plen = int(plen_all[s + j]); r = rng.uniform(a.r_lo, a.r_hi); new = x.copy(); sub = np.zeros(len(x), bool); last = -10 ** 9
            for t in range(max(plen, h), len(x)):
                if t - last > h and rng.random() < r:
                    p = torch.softmax(logits[j, t - 1, :V].float(), -1); p[int(x[t])] = 0
                    if p.sum() <= 0: continue
                    new[t] = int(torch.multinomial(p / p.sum(), 1, generator=gen)); sub[t] = True; last = t
            piv, n, dup = wm.score(new, start=plen)
            seen = set(); cat = []
            for t in range(max(plen, h), len(new)):
                ctx = tuple(int(v) for v in new[t - h:t])
                if ctx in seen: continue
                seen.add(ctx)
                ctx_ok = not sub[t - h:t].any()
                cat.append(2 if (ctx_ok and sub[t]) else (1 if (ctx_ok and not sub[t]) else 0))
            cat = np.array(cat)
            piv_out.append(piv.astype(np.float32)); pv_out.append(doc_pvalue(piv, scheme, rng=rng)); ns.append(n); rates.append(r)
            eps_eff.append(float((cat == 1).mean()) if len(cat) else np.nan); sig_eff.append(float((cat == 2).mean()) if len(cat) else np.nan)
        if s % 400 == 0: print(s, flush=True)
    out = a.npz.replace(".npz", "_subst.npz"); pv = np.array(pv_out)
    np.savez(out, pivots=np.array(piv_out, dtype=object), pvals=pv, n=np.array(ns), eps_eff=np.array(eps_eff), sig_eff=np.array(sig_eff), rate=np.array(rates))
    print({"out": out, "docs": len(pv), "frac_p_le_0.05": round(float((pv <= 0.05).mean()), 3), "eps_eff_mean": round(float(np.nanmean(eps_eff)), 3),
           "sig_eff_mean": round(float(np.nanmean(sig_eff)), 3), "sig_eff_q95": round(float(np.nanquantile(sig_eff, .95)), 3), "median_n": float(np.median(ns))})

if __name__ == "__main__":
    main()
