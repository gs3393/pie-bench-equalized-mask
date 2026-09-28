#!/usr/bin/env bash
WS="${PIE_BENCH_WORKSPACE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$WS"
export HF_HOME="${HF_HOME:-$WS/.cache/huggingface}" CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHONPATH=$WS/baselines/FlowAlign ./.venv/bin/python runner/run_flowalign.py --src_path PIE-bench-20 --saved_path outputs/flowalign1024_nomask --mask_dir none --img_shape 1024 && echo "GEN_DONE"
SUBSET=$WS/PIE-bench-20 ./runner/eval_all.sh flowalign1024_nomask
echo "RESCHECK_DONE"
