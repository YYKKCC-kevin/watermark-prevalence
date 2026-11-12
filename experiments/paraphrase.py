"""LLM paraphrase of a subset of watermarked docs (assumption-deviation experiment).
Paraphraser: Qwen/Qwen2.5-3B-Instruct (cached).  The paraphrased text is re-tokenized with the WATERMARKED model's
tokenizer and scored under the key.  Realized retention is unknown here by construction.
Usage: CUDA_VISIBLE_DEVICES=2 python3 paraphrase.py --npz data/gen/Qwen3-1p7B_gumbel.npz --n 300
"""
import argparse, numpy as np, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from wm import Watermarker, doc_pvalue

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--paraphraser", default="Qwen/Qwen2.5-3B-Instruct"); ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args(); torch.set_grad_enabled(False)
    d = np.load(a.npz, allow_pickle=True); scheme = str(d["scheme"]); key = int(d["key"]); h = int(d["h"]); V = int(d["V"]); model_name = str(d["model"])
    wtok = AutoTokenizer.from_pretrained(model_name); wm = Watermarker(scheme, key, V, h=h)
    ptok = AutoTokenizer.from_pretrained(a.paraphraser); ptok.padding_side = "left"
    pm = AutoModelForCausalLM.from_pretrained(a.paraphraser, dtype=torch.float16).cuda().eval()
    ids_all, plen_all = d["ids"][: a.n], d["prompt_len"][: a.n]
    texts = [wtok.decode(np.asarray(x)[p:].tolist(), skip_special_tokens=True) for x, p in zip(ids_all, plen_all)]
    rng = np.random.default_rng(1); piv_out, pv_out, ns, ratio = [], [], [], []; t0 = time.time()
    for s in range(0, len(texts), a.batch):
        msgs = [[{"role": "system", "content": "You are a careful editor."},
                 {"role": "user", "content": "Paraphrase the following text sentence by sentence. Keep the meaning and roughly the same length. Output only the paraphrase.\n\n" + t}] for t in texts[s:s + a.batch]]
        # ChatML written out by hand (Qwen2.5 format); apply_chat_template needs jinja2>=3.1, which this env lacks
        prompts = ["".join(f"<|im_start|>{x['role']}\n{x['content']}<|im_end|>\n" for x in m) + "<|im_start|>assistant\n" for m in msgs]
        enc = ptok(prompts, return_tensors="pt", padding=True).to("cuda")
        out = pm.generate(**enc, max_new_tokens=400, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=ptok.pad_token_id)
        for j in range(out.shape[0]):
            para = ptok.decode(out[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
            ids = wtok(para).input_ids
            piv, n, dup = wm.score(ids, start=0)
            if n < 20: continue
            piv_out.append(piv.astype(np.float32)); pv_out.append(doc_pvalue(piv, scheme, rng=rng)); ns.append(n); ratio.append(len(ids) / max(1, len(ids_all[s + j]) - plen_all[s + j]))
        print(s + a.batch, f"{time.time()-t0:.0f}s", flush=True)
    out = a.npz.replace(".npz", "_paraphrase.npz"); pv = np.array(pv_out)
    np.savez(out, pivots=np.array(piv_out, dtype=object), pvals=pv, n=np.array(ns), len_ratio=np.array(ratio))
    print({"out": out, "docs": len(pv), "frac_p_le_0.05": round(float((pv <= 0.05).mean()), 3), "frac_p_le_0.01": round(float((pv <= 0.01).mean()), 3), "median_n": float(np.median(ns)), "median_len_ratio": round(float(np.median(ratio)), 2)})

if __name__ == "__main__":
    main()
