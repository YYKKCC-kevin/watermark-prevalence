"""Score human (C4) documents under the watermark key: null p-values and the (S0) uniformity check.
Usage: HF_HOME=<cache dir> python3 score_human.py --model facebook/opt-1.3b --scheme gumbel --n_docs 4000 --skip 2600
Human docs are taken AFTER the first 2600 C4 docs (generation prompts use the first ~2600), so no overlap.
"""
import argparse, os, time, numpy as np
from transformers import AutoTokenizer, AutoConfig
from scipy import stats
from wm import Watermarker, doc_pvalue
from gen import load_c4

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--scheme", required=True)
    ap.add_argument("--n_docs", type=int, default=4000); ap.add_argument("--skip", type=int, default=2600)
    ap.add_argument("--max_tokens", type=int, default=330); ap.add_argument("--key", type=int, default=20261111); ap.add_argument("--h", type=int, default=3)
    ap.add_argument("--c4", default="data/c4/en/c4-validation.00000-of-00008.json.gz"); ap.add_argument("--out", default="data/gen")
    a = ap.parse_args(); tag = a.model.split("/")[-1].replace(".", "p"); os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model); V = AutoConfig.from_pretrained(a.model).vocab_size
    wm = Watermarker(a.scheme, a.key, V, h=a.h)
    docs = load_c4(a.c4, a.n_docs, skip=a.skip)
    rng = np.random.default_rng(a.key + 1); t0 = time.time()
    piv_all, pv_all, dups, ns = [], [], [], []
    for k, d in enumerate(docs):
        ids = tok(d, truncation=True, max_length=a.max_tokens).input_ids
        piv, n, dup = wm.score(ids, start=30)
        if n < 50: continue
        piv_all.append(piv.astype(np.float32)); pv_all.append(doc_pvalue(piv, a.scheme, rng=rng)); dups.append(dup); ns.append(n)
        if k % 500 == 0: print(k, f"{time.time()-t0:.0f}s", flush=True)
    pv = np.array(pv_all); allpiv = np.concatenate(piv_all)
    ks_doc = stats.kstest(pv, "uniform")
    rep = {"model": a.model, "scheme": a.scheme, "docs": len(pv), "median_n": float(np.median(ns)), "dup_frac": float(np.sum(dups) / (np.sum(dups) + np.sum(ns))),
           "doc_p_KS": [float(ks_doc.statistic), float(ks_doc.pvalue)], "frac_p_le_0.05": float((pv <= 0.05).mean()), "frac_p_le_0.01": float((pv <= 0.01).mean())}
    if a.scheme == "gumbel":
        ks_piv = stats.kstest(allpiv, "uniform"); rep["pivot_KS"] = [float(ks_piv.statistic), float(ks_piv.pvalue)]; rep["pivot_mean"] = float(allpiv.mean())
    else:
        rep["green_frac"] = float(allpiv.mean()); rep["green_frac_se"] = float(np.sqrt(0.25 / len(allpiv)))
    print(rep)
    np.savez(os.path.join(a.out, f"{tag}_{a.scheme}_human.npz"), pivots=np.array(piv_all, dtype=object), pvals=pv, n=np.array(ns), report=str(rep))

if __name__ == "__main__":
    main()
