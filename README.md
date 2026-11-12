# Code for "Prevalence Bounds for Edited Watermarked Text: Power Envelopes and Calibration Sensitivity"

Assumption-indexed bounds on the share of watermarked documents in a corpus under unknown edits, with exact or reference-calibrated nulls.

Kuancheng Ye, Department of Statistics, University of California, Irvine.

## Layout
- `sim/` — theory computations and simulations (CPU only).
  - `envelope.py`, `certified.py`: Gumbel-max power envelope and its conservative lattice/FFT lower bound (`certified_envelope(n, eps_min, delta, ts, sigma=...)`).
  - `kgw.py`: KGW closed-form envelope (`h_kgw(..., sigma=...)`).
  - `substitution.py`: substituted-position laws (Gumbel `F_sub`, KGW floor) with Monte Carlo / enumeration checks.
  - `betting_cs.py`: betting confidence sequence and realized-fraction confidence sequence.
  - `realized.py`: realized-share bounds with the exact null (Prop. realized) and with a reference-calibrated null band (Prop. calibrated).
  - `weak_signal.py`: weak-signal simulations and fair single-threshold baselines; `null_sensitivity.py`: sensitivity to null miscalibration.
  - `check_fft_actual_lengths.py`: FFT vs direct summation and lattice refinement at the lengths used.
  - `grid_certified.py`, `outclass.py`, `lp_diag.py`, `dgp.py`: simulations; `check_fft_error.py`, `check_envelope_monotone.py`: numerical checks.
- `experiments/` — real-model pipeline (one GPU).
  - `wm.py` (Gumbel-max and KGW watermarking, scoring, deduplication, p-values), `gen.py` (generation), `score_human.py` (human C4 scoring), `edits.py` (random and near-floor/short edits), `subst.py` (plausible substitutions), `paraphrase.py` (LLM paraphrase), `analysis.py` (fixed-composition design), `analysis_stage4.py` (normalized class check, per-token diagnostics), `plots.py`.
  - `run_calib.sh` (fresh human documents under three keys, new-key generations), `calib_analysis.py` (per-key KS, held-out coverage), `calib_crosscheck.py` (old vs fresh human batch), `realized_perturb.py` (main real-data design: realized share, exact vs calibrated null).
- `results/` — all result JSONs used in the paper (`sim/` and `real/`).

## Reproduce
Simulations: `cd sim && python3 -I certified.py --out ../results/sim` (and the other scripts; each documents its command in the header).
Real models (set `HF_HOME`; C4 validation shard 0 under `experiments/data/c4/`):
1. `python3 gen.py --model facebook/opt-1.3b --scheme gumbel` (and `kgw`; and `--model Qwen/Qwen3-1.7B`).
2. `python3 score_human.py --model ... --scheme ...`.
3. `python3 edits.py --npz data/gen/<tag>_<scheme>.npz --mode scattered --eps_lo 0.3` (also `span`, `--eps_hi 0.4`, `--max_new 100`), `python3 subst.py --npz ...`, `python3 paraphrase.py --npz ... --n 320`.
4. `PYTHONPATH=.:../sim python3 analysis_stage4.py --tag <tag> --scheme <scheme> [--null pool] [--extra_pools ...]`.
5. Calibration: `bash run_calib.sh`, then `PYTHONPATH=.:../sim python3 calib_analysis.py --tag <tag> --scheme <scheme>` and `calib_crosscheck.py` likewise.
6. Main real-data table: `PYTHONPATH=.:../sim python3 realized_perturb.py --tag <tag> --scheme <scheme>` (needs the fresh human pool under key 20261111).
Seeds are fixed in each script. Watermark keys: 20261111 (main), 20261112 and 20261113 (calibration); context window 3.

## Versions
See `versions.txt` (torch, transformers, numpy, scipy).

## Citation
If you use this code, please cite:

```
@misc{ye2026prevalence,
  title  = {Prevalence Bounds for Edited Watermarked Text: Power Envelopes and Calibration Sensitivity},
  author = {Ye, Kuancheng},
  year   = {2026},
  note   = {arXiv preprint}
}
```
