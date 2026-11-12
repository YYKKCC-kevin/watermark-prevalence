"""Generate watermarked completions from C4 prompts and score them.
Usage: HF_HOME=<cache dir> python3 gen.py --model facebook/opt-1.3b --scheme gumbel --n_docs 2000
Outputs data/gen/<model_tag>_<scheme>.npz: ids, pivots, maxp (object arrays), prompt_len, pvals.
"""
import argparse, gzip, json, os, time, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from wm import Watermarker, doc_pvalue

def load_c4(path, n, min_chars=800, skip=0):
    docs = []
    with gzip.open(path, "rt") as f:
        for line in f:
            t = json.loads(line)["text"]
            if len(t) >= min_chars:
                if skip > 0: skip -= 1; continue
                docs.append(t)
                if len(docs) >= n: break
    return docs

def vocab_size(model):
    return model.get_input_embeddings().weight.shape[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--scheme", required=True)
    ap.add_argument("--n_docs", type=int, default=2000); ap.add_argument("--prompt_tokens", type=int, default=30)
    ap.add_argument("--max_new", type=int, default=300); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--key", type=int, default=20261111); ap.add_argument("--h", type=int, default=3)
    ap.add_argument("--c4", default="data/c4/en/c4-validation.00000-of-00008.json.gz"); ap.add_argument("--out", default="data/gen")
    ap.add_argument("--min_new", type=int, default=50)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); torch.set_grad_enabled(False)
    tag = a.model.split("/")[-1].replace(".", "p")
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).cuda().eval()
    V = vocab_size(model)
    wm = Watermarker(a.scheme, a.key, V, h=a.h)
    docs = load_c4(a.c4, int(a.n_docs * 1.3))
    ids_all, piv_all, maxp_all, plen_all, pv_all = [], [], [], [], []
    rng = np.random.default_rng(a.key); t0 = time.time(); i = 0
    while len(ids_all) < a.n_docs and i < len(docs):
        batch = docs[i:i + a.batch]; i += a.batch
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=a.prompt_tokens).to("cuda")
        input_ids, attn = enc.input_ids, enc.attention_mask
        B = input_ids.shape[0]
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
        past = out.past_key_values; logits = out.logits[:, -1, :]
        gen = [[] for _ in range(B)]; maxps = [[] for _ in range(B)]
        done = torch.zeros(B, dtype=torch.bool, device="cuda")
        ctx_ids = [input_ids[j][attn[j].bool()].tolist() for j in range(B)]
        for step in range(a.max_new):
            ctxs = [tuple(ctx_ids[j][-a.h:]) for j in range(B)]
            toks, piv, maxp = wm.step(logits, ctxs)
            tl = toks.tolist(); ml = maxp.tolist(); dl = done.tolist()
            for j in range(B):
                if not dl[j]:
                    gen[j].append(tl[j]); maxps[j].append(ml[j]); ctx_ids[j].append(tl[j])
            done |= toks == tok.eos_token_id
            if done.all(): break
            attn = torch.cat([attn, torch.ones(B, 1, dtype=attn.dtype, device="cuda")], 1)
            out = model(input_ids=toks[:, None], attention_mask=attn, past_key_values=past, use_cache=True)
            past = out.past_key_values; logits = out.logits[:, -1, :]
            if os.environ.get("GEN_DEBUG") and step % 25 == 0:
                print(f"  step {step}: alloc {torch.cuda.memory_allocated()/2**30:.2f} GiB, reserved {torch.cuda.memory_reserved()/2**30:.2f} GiB, attn {tuple(attn.shape)}, logits {tuple(out.logits.shape)}", flush=True)
        for j in range(B):
            g = gen[j]
            if g and g[-1] == tok.eos_token_id: g = g[:-1]
            if len(g) < a.min_new: continue
            prompt = input_ids[j][attn[j, :input_ids.shape[1]].bool()].tolist()
            full = prompt + g; plen = len(prompt)
            piv_s, n, dup = wm.score(full, start=plen)
            ids_all.append(np.array(full, dtype=np.int32)); piv_all.append(piv_s.astype(np.float32))
            maxp_all.append(np.array(maxps[j][:len(g)], dtype=np.float32)); plen_all.append(plen)
            pv_all.append(doc_pvalue(piv_s, a.scheme, rng=rng))
        print(f"{len(ids_all)} docs, {time.time()-t0:.0f}s, batch mean p={np.mean(pv_all[-B:]):.3g}, median n={np.median([len(p) for p in piv_all[-B:]]):.0f}", flush=True)
    np.savez(os.path.join(a.out, f"{tag}_{a.scheme}.npz"), ids=np.array(ids_all, dtype=object), pivots=np.array(piv_all, dtype=object),
             maxp=np.array(maxp_all, dtype=object), prompt_len=np.array(plen_all), pvals=np.array(pv_all), key=a.key, h=a.h, scheme=a.scheme, model=a.model, V=V)
    print("saved", len(ids_all), "docs; median n =", int(np.median([len(p) for p in piv_all])), "; frac p<0.05 =", float(np.mean(np.array(pv_all) < 0.05)))

if __name__ == "__main__":
    main()
