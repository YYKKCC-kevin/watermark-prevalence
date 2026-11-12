#!/bin/bash
# Calibration study (Stage 4 of the must-do list, 2026-11-12): three keys x two pools (Qwen3/KGW, OPT/Gumbel).
# Human: 12000 fresh C4 documents (eligible docs 6600-18599) scored under each key.  Watermarked: 1000 docs per new key,
# then scattered edits eps ~ U[0.3,1].  Original key 20261111 reuses its existing 2000-doc generations and edits.
cd "$(dirname "$0")"; export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} PYTHONPATH=$PWD:$PWD/../sim CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
L=data/results/calib_driver.log; : > $L
declare -A MODEL=( [Qwen3-1p7B]=Qwen/Qwen3-1.7B [opt-1p3b]=facebook/opt-1.3b ); declare -A SCH=( [Qwen3-1p7B]=kgw [opt-1p3b]=gumbel )
for k in 20261111 20261112 20261113; do mkdir -p data/calib/k$k; done
for tag in Qwen3-1p7B opt-1p3b; do for k in 20261111 20261112 20261113; do
  ( python3 score_human.py --model ${MODEL[$tag]} --scheme ${SCH[$tag]} --key $k --skip 6600 --n_docs 12000 --out data/calib/k$k > data/calib/k$k/human_${tag}.log 2>&1; echo "human $tag $k exit $? $(date +%T)" >> $L ) &
done; done
for tag in Qwen3-1p7B opt-1p3b; do for k in 20261112 20261113; do
  ( python3 gen.py --model ${MODEL[$tag]} --scheme ${SCH[$tag]} --key $k --n_docs 1000 --out data/calib/k$k > data/calib/k$k/gen_${tag}.log 2>&1 && \
    python3 edits.py --npz data/calib/k$k/${tag}_${SCH[$tag]}.npz --mode scattered --eps_lo 0.3 > data/calib/k$k/edit_${tag}.log 2>&1; echo "gen+edit $tag $k exit $? $(date +%T)" >> $L ) &
done; done
wait; echo "CALIB_DONE $(date)" >> $L
