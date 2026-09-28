#!/usr/bin/env bash
# 생성이 끝난 뒤: 픽셀 합성 조건 생성 -> 전 조건 평가 -> 요약표
set -e
WS=~/workspace/pie-bench; cd $WS
export HF_HOME=~/.cache/huggingface CUDA_VISIBLE_DEVICES=${GPU:-1}
for m in ftedit flowedit flowalign dnaedit; do
  ./.venv-data/bin/python runner/pixel_paste.py --subset PIE-bench-150 --in_dir outputs/${m}_nomask --out_dir outputs/${m}_pixpaste
done
CONDS="directedit_pixpaste"
for m in ftedit flowedit flowalign dnaedit; do CONDS="$CONDS ${m}_nomask ${m}_gtmask ${m}_pixpaste"; done
./runner/eval_all.sh $CONDS
./.venv/bin/python runner/summarize_multi.py > results/summary_multi.txt
echo "FINISH_DONE"
