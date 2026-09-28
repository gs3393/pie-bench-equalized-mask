"""블렌딩용 마스크 두 벌을 만든다.

gt   : PIE-Bench GT 마스크를 그대로 PNG로 저장. 평가 마스크와 동일하다.
auto : GT 마스크에서 유도한 bbox로 SAM2를 돌린 근사 마스크.
       논문은 MLLM(Qwen3-VL)로 편집 유형과 bbox를 정하지만 API 키가 없어
       벤치마크의 카테고리 라벨과 GT bbox로 대체했다. 논문 마스크보다
       유리한(낙관적인) 대용물이다.

규칙
  - 테두리 강제 1픽셀을 제거한 뒤 면적 > 0.95 -> 전체 마스크 (Global 편집)
  - editing_type_id == "8" (배경 변경) -> 여집합의 bbox로 SAM2 후 반전
  - 그 외 -> 마스크의 bbox로 SAM2
  - dilation kernel 8 (저장소 generate_mask.py 기본값)
"""
import argparse, json, os
import numpy as np, cv2, torch
from PIL import Image

SAM2_ID = "facebook/sam2.1-hiera-large"


def decode(enc, shape=(512, 512)):
    L = shape[0] * shape[1]
    a = np.zeros(L, np.uint8)
    for i in range(0, len(enc), 2):
        st, ln = enc[i], min(enc[i + 1], L - enc[i])
        a[st:st + ln] = 1
    return a.reshape(shape)


def strip_border(m):
    m = m.copy()
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 0
    return m


def bbox_of(m):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--gt_out", required=True)
    ap.add_argument("--auto_out", required=True)
    ap.add_argument("--dilate", type=int, default=8)
    ap.add_argument("--global_thresh", type=float, default=0.95)
    args = ap.parse_args()

    from transformers import Sam2Model, Sam2Processor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam2Model.from_pretrained(SAM2_ID, dtype=torch.float16).to(dev).eval()
    proc = Sam2Processor.from_pretrained(SAM2_ID)

    mapping = json.load(open(os.path.join(args.src, "mapping_file.json")))
    os.makedirs(args.gt_out, exist_ok=True)
    os.makedirs(args.auto_out, exist_ok=True)

    stats = {"global": 0, "background": 0, "object": 0, "empty": 0}
    ious = []
    for key, item in sorted(mapping.items()):
        img_path = os.path.join(args.src, "annotation_images", item["image_path"])
        stem = os.path.splitext(os.path.basename(item["image_path"]))[0]
        gt = decode(item["mask"])
        cv2.imwrite(os.path.join(args.gt_out, stem + "_mask.png"), gt * 255)

        core = strip_border(gt)
        area = core.mean()
        if area > args.global_thresh:
            auto = np.ones_like(core) * 255
            stats["global"] += 1
        else:
            invert = item["editing_type_id"] == "8"
            target = (1 - core) if invert else core
            target = strip_border(target) if invert else target
            box = bbox_of(target)
            if box is None:
                auto = np.ones_like(core) * 255
                stats["empty"] += 1
            else:
                rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                inputs = proc(images=Image.fromarray(rgb), input_boxes=[[box.tolist()]], return_tensors="pt")
                sizes = inputs["original_sizes"]
                moved = {}
                for k, v in inputs.items():
                    if not torch.is_tensor(v):
                        moved[k] = v
                    elif torch.is_floating_point(v):
                        moved[k] = v.to(device=dev, dtype=torch.float16)
                    else:
                        moved[k] = v.to(dev)
                with torch.no_grad():
                    out = model(**moved, multimask_output=False)
                masks = proc.post_process_masks(out.pred_masks.cpu(), sizes)[0]
                sam = (masks[0, 0] if masks.ndim == 4 else masks[0]).numpy() > 0
                sam = sam.astype(np.uint8)
                if invert:
                    sam = 1 - sam
                    stats["background"] += 1
                else:
                    stats["object"] += 1
                auto = sam * 255
                if args.dilate > 0:
                    auto = cv2.dilate(auto, np.ones((args.dilate, args.dilate), np.uint8), iterations=1)
        cv2.imwrite(os.path.join(args.auto_out, stem + "_mask.png"), auto)

        a = (auto > 127).astype(np.uint8)
        inter = (a & core).sum()
        union = (a | core).sum()
        if union > 0:
            ious.append(inter / union)

    print("counts:", stats)
    print("auto vs GT IoU: mean %.3f  median %.3f  n=%d" % (np.mean(ious), np.median(ious), len(ious)))
    print("auto mean area %.3f" % np.mean([ (cv2.imread(os.path.join(args.auto_out,f),0)>127).mean() for f in sorted(os.listdir(args.auto_out)) ]))


if __name__ == "__main__":
    main()
