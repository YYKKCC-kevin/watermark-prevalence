#!/bin/bash
# Runs after the four generation jobs and four human-scoring jobs finish: edits -> paraphrase -> analysis.
cd .
export CUDA_VISIBLE_DEVICES=2 PYTHONPATH=.:../sim
until [ $(grep -l "^saved" data/gen/logs/gen_*.log 2>/dev/null | wc -l) -eq 4 ] && [ $(grep -l "{'model'" data/gen/logs/human_*.log 2>/dev/null | wc -l) -eq 4 ]; do sleep 30; done
echo "GEN_DONE $(date)"
for tag in opt-1p3b Qwen3-1p7B; do for s in gumbel kgw; do
  for mode in scattered span; do for lo in 0.3 0.5; do
    python3 edits.py --npz data/gen/${tag}_${s}.npz --mode $mode --eps_lo $lo 2>&1 | tail -1
  done; done
done; done
echo "EDITS_DONE $(date)"
for tag in opt-1p3b Qwen3-1p7B; do for s in gumbel kgw; do
  python3 paraphrase.py --npz data/gen/${tag}_${s}.npz --n 320 2>&1 | grep -v -i warning | tail -1
done; done
echo "PARA_DONE $(date)"
for tag in opt-1p3b Qwen3-1p7B; do for s in gumbel kgw; do for lo in 0.3 0.5; do
  python3 analysis.py --tag $tag --scheme $s --edit_eps_lo $lo --R 50 --N 4000 > data/results/log_${tag}_${s}_lo${lo}.txt 2>&1
  echo "analysis $tag $s $lo exit $?"
done; done; done
echo "ALL_DONE $(date)"
