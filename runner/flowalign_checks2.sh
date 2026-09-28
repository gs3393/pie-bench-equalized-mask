#!/usr/bin/env bash
# 감사 후 재실험: (a) 1024² 를 원본 크기로 되돌려 저장한 뒤 채점, (b) 상류와 같은 min/max 저장 정규화의 효과. 20장, GPU 1.
WS=~/workspace/pie-bench; cd $WS
export HF_HOME=~/.cache/huggingface CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=$WS/.venv/bin/python
PYTHONPATH=$WS/baselines/FlowAlign $P runner/run_flowalign.py --src_path PIE-bench-20 --saved_path outputs/flowalign1024ds_nomask --mask_dir none --img_shape 1024 && echo "GEN1_DONE"
PYTHONPATH=$WS/baselines/FlowAlign $P runner/run_flowalign.py --src_path PIE-bench-20 --saved_path outputs/flowalign_minmax_nomask --mask_dir none --img_shape 512 --save_norm minmax && echo "GEN2_DONE"
SUBSET=$WS/PIE-bench-20 ./runner/eval_all.sh flowalign1024ds_nomask flowalign_minmax_nomask
$P runner/rescheck_compare.py flowalign1024ds_nomask flowalign_nomask flowalign_minmax_nomask flowalign1024_nomask > results/rescheck2_flowalign.txt 2>&1
echo "CHECKS2_DONE"
