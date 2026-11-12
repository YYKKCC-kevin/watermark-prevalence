#!/bin/bash
# Rerun after fixes (2026-11-12): paraphrase x4 (hand-written ChatML), then analysis x8 and plots.
cd "$(dirname "$0")"; export PYTHONPATH=.:../sim CUDA_VISIBLE_DEVICES=2
L=data/results/driver2.log; : > $L
for tag in opt-1p3b Qwen3-1p7B; do for s in gumbel kgw; do
  python3 paraphrase.py --npz data/gen/${tag}_${s}.npz --n 320 2>&1 | grep -v -i warn | tail -1 >> $L
done; done
echo "PARA_DONE $(date)" >> $L
for tag in opt-1p3b Qwen3-1p7B; do for s in gumbel kgw; do for lo in 0.3 0.5; do
  python3 analysis.py --tag $tag --scheme $s --edit_eps_lo $lo > data/results/log_${tag}_${s}_lo${lo}.txt 2>&1; echo "analysis $tag $s $lo exit $?" >> $L
done; done; done
python3 plots.py --res data/results >> $L 2>&1; echo "plots exit $?" >> $L
echo "ALL_DONE $(date)" >> $L
