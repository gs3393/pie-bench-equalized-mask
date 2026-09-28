#!/usr/bin/env bash
WS="${PIE_BENCH_WORKSPACE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$WS"
export HF_HOME="${HF_HOME:-$WS/.cache/huggingface}" CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$WS/PIE-bench-150; P=$WS/.venv/bin/python
PYTHONPATH=$WS/baselines/FTEdit $P runner/run_ftedit.py  --src_path $S --saved_path outputs/ftedit_nomask  --mask_dir none        && echo "COND_DONE ftedit_nomask"
PYTHONPATH=$WS/baselines/FTEdit $P runner/run_ftedit.py  --src_path $S --saved_path outputs/ftedit_gtmask  --mask_dir $S/mask_gt  && echo "COND_DONE ftedit_gtmask"
$P runner/run_dnaedit.py --src_path $S --saved_path outputs/dnaedit_nomask --mask_dir none        && echo "COND_DONE dnaedit_nomask"
$P runner/run_dnaedit.py --src_path $S --saved_path outputs/dnaedit_gtmask --mask_dir $S/mask_gt  && echo "COND_DONE dnaedit_gtmask"
echo "QUEUE_DONE gpu1"
