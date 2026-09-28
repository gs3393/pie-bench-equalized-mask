#!/usr/bin/env bash
set -x
WS="${PIE_BENCH_WORKSPACE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$WS"
export HF_HOME="${HF_HOME:-$WS/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$WS/DirectEdit
S=$WS/PIE-bench-150
P=$WS/.venv/bin/python

$P runner/run_condition.py --src_path $S --saved_path outputs/nomask   --mask_dir none                && echo "COND_DONE nomask"
$P runner/run_condition.py --src_path $S --saved_path outputs/automask --mask_dir $S/mask_auto        && echo "COND_DONE automask"
$P runner/run_condition.py --src_path $S --saved_path outputs/gtmask   --mask_dir $S/mask_gt          && echo "COND_DONE gtmask"
$P runner/vae_roundtrip.py --src_path $S --saved_path outputs/vaebound                                && echo "COND_DONE vaebound"
echo "ALL_GEN_DONE"
