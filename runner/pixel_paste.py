"""픽셀 공간 합성 조건: 어떤 방법의 결과든 GT 마스크 밖을 원본 픽셀로 덮는다.

배경 열의 바닥선 통제군이다. GPU 가 필요 없다.
입력: <in_dir>/<stem>_edited.png (원본 형식: [재구성|편집] 가로 결합 또는 단일 512²)
출력: <out_dir>/<stem>_edited.png (같은 형식으로 저장해 evaluate.py 가 그대로 채점)
마스크: 평가와 같은 PIE-Bench GT (mapping_file.json 의 RLE, 테두리 1 강제). 팽창 없음.
"""
import argparse, json, os, numpy as np
from PIL import Image


def decode(enc, shape=(512, 512)):
    L = shape[0] * shape[1]; a = np.zeros(L, np.uint8)
    for i in range(0, len(enc), 2):
        st, ln = enc[i], min(enc[i + 1], L - enc[i]); a[st:st + ln] = 1
    a = a.reshape(shape); a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = 1
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    m = json.load(open(os.path.join(a.subset, "mapping_file.json")))
    os.makedirs(a.out_dir, exist_ok=True)
    n = 0
    for k, it in sorted(m.items()):
        stem = os.path.splitext(os.path.basename(it["image_path"]))[0]
        src = np.asarray(Image.open(os.path.join(a.subset, "annotation_images", it["image_path"])).convert("RGB"))
        g = Image.open(os.path.join(a.in_dir, stem + "_edited.png")).convert("RGB")
        W, H = g.size
        tar = np.asarray(g.crop((W - 512, H - 512, W, H)).resize((512, 512)))
        mk = decode(it["mask"])[:, :, None]
        out = (tar * mk + src * (1 - mk)).astype(np.uint8)
        canvas = np.asarray(g).copy()
        canvas[H - 512:, W - 512:] = out
        Image.fromarray(canvas).save(os.path.join(a.out_dir, stem + "_edited.png"))
        n += 1
    print("pixel-pasted", n, "->", a.out_dir)


if __name__ == "__main__":
    main()
