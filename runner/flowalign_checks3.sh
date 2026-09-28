#!/usr/bin/env bash
# 저장 정규화 가설 확인: (a) 151장 512² min/max 저장 (GPU1), (b) 20장 1024²+min/max 저장 (GPU0). 병렬.
WS="${PIE_BENCH_WORKSPACE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$WS"
export HF_HOME="${HF_HOME:-$WS/.cache/huggingface}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=$WS/.venv/bin/python
( CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$WS/baselines/FlowAlign $P runner/run_flowalign.py --src_path PIE-bench-150 --saved_path outputs/flowalign_minmax151_nomask --mask_dir none --img_shape 512 --save_norm minmax > logs_c3a.txt 2>&1 && echo "GENA_DONE" >> logs_c3a.txt ) &
( CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$WS/baselines/FlowAlign $P runner/run_flowalign.py --src_path PIE-bench-20 --saved_path outputs/flowalign1024ds_minmax_nomask --mask_dir none --img_shape 1024 --save_norm minmax > logs_c3b.txt 2>&1 && echo "GENB_DONE" >> logs_c3b.txt ) &
wait
CUDA_VISIBLE_DEVICES=1 SUBSET=$WS/PIE-bench-150 ./runner/eval_all.sh flowalign_minmax151_nomask
CUDA_VISIBLE_DEVICES=1 SUBSET=$WS/PIE-bench-20 ./runner/eval_all.sh flowalign1024ds_minmax_nomask
$P runner/rescheck_compare.py flowalign_minmax151_nomask flowalign_nomask > results/rescheck3_151.txt 2>&1
$P runner/rescheck_compare.py flowalign1024ds_minmax_nomask flowalign_nomask flowalign_minmax_nomask flowalign1024ds_nomask > results/rescheck3_20.txt 2>&1
echo "CHECKS3_DONE"
