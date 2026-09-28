#!/usr/bin/env bash
# 조건별로 DirectEdit 평가 스크립트를 그대로 돌린다.
# evaluate.py 가 stat_ 파일을 CWD 상대경로로 쓰기 때문에 evaluation/ 에서 돌린 뒤 옮긴다.
set -e
WS=~/workspace/pie-bench
SUBSET=${SUBSET:-$WS/PIE-bench-150}
export PYTHONPATH=$WS/DirectEdit:$PYTHONPATH
mkdir -p $WS/results
cd $WS/DirectEdit/evaluation
for c in "$@"; do
  echo "=== evaluating $c ==="
  $WS/.venv/bin/python evaluate.py \
    --metrics structure_distance psnr_unedit_part lpips_unedit_part mse_unedit_part ssim_unedit_part \
              clip_similarity_source_image clip_similarity_target_image clip_similarity_target_image_edit_part \
    --result_path $c.csv \
    --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
    --tar_image_folder $WS/outputs/$c \
    --tar_method $c \
    --src_image_folder $SUBSET/annotation_images \
    --annotation_mapping_file $SUBSET/mapping_file.json > $WS/results/$c.log 2>&1
  mv $c.csv stat_$c.csv $WS/results/
  echo "  -> $WS/results/$c.csv"
done
