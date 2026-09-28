#!/usr/bin/env bash
WS="${PIE_BENCH_WORKSPACE:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"; cd "$WS"
export HF_HOME="${HF_HOME:-$WS/.cache/huggingface}" CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$WS/PIE-bench-150; P=$WS/.venv/bin/python
PYTHONPATH=$WS/baselines/FlowEdit-vanilla $P runner/run_flowedit.py  --src_path $S --saved_path outputs/flowedit_nomask  --mask_dir none        && echo "COND_DONE flowedit_nomask"
PYTHONPATH=$WS/baselines/FlowEdit-vanilla $P runner/run_flowedit.py  --src_path $S --saved_path outputs/flowedit_gtmask  --mask_dir $S/mask_gt  && echo "COND_DONE flowedit_gtmask"
PYTHONPATH=$WS/baselines/FlowAlign       $P runner/run_flowalign.py --src_path $S --saved_path outputs/flowalign_nomask --mask_dir none        && echo "COND_DONE flowalign_nomask"
PYTHONPATH=$WS/baselines/FlowAlign       $P runner/run_flowalign.py --src_path $S --saved_path outputs/flowalign_gtmask --mask_dir $S/mask_gt  && echo "COND_DONE flowalign_gtmask"
echo "QUEUE_DONE gpu0"
