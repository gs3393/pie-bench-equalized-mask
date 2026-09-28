"""PIE-Bench의 psnr_unedit_part 와 "배경 픽셀만" PSNR 을 함께 잰다.

psnr_unedit_part 는 두 이미지에서 편집 영역을 0으로 지운 뒤 이미지 전체로 PSNR 을
계산한다. 오차가 0인 픽셀이 분모에 함께 들어가므로 편집 마스크가 클수록 값이 커진다.
비교용으로 배경 픽셀만 평균한 PSNR 도 같이 낸다.
"""
import json, os, sys, numpy as np
from PIL import Image
S = "PIE-bench-150"
CONDS = sys.argv[1:]
m = json.load(open(S + "/mapping_file.json"))

def decode(enc, shape=(512, 512)):
    L = shape[0] * shape[1]; a = np.zeros(L, np.uint8)
    for i in range(0, len(enc), 2):
        st, ln = enc[i], min(enc[i + 1], L - enc[i]); a[st:st + ln] = 1
    a = a.reshape(shape); a[0, :] = a[-1, :] = a[:, 0] = a[:, -1] = 1
    return a

rows = {c: {"whole": [], "bgonly": [], "area": []} for c in CONDS}
for k, it in sorted(m.items()):
    stem = os.path.splitext(os.path.basename(it["image_path"]))[0]
    bg = 1 - decode(it["mask"])
    if bg.sum() == 0:
        continue
    src = np.asarray(Image.open(os.path.join(S, "annotation_images", it["image_path"])).convert("RGB"), np.float32) / 255
    bg3 = bg[:, :, None].astype(np.float32)
    for c in CONDS:
        p = "outputs/%s/%s_edited.png" % (c, stem)
        g = Image.open(p).convert("RGB")
        tar = np.asarray(g.crop((g.size[0] - 512, g.size[1] - 512, g.size[0], g.size[1])).resize((512, 512)), np.float32) / 255
        d2 = (src * bg3 - tar * bg3) ** 2
        whole = 10 * np.log10(1.0 / d2.mean())
        bgonly = 10 * np.log10(1.0 / (d2.sum() / (bg.sum() * 3)))
        rows[c]["whole"].append(whole); rows[c]["bgonly"].append(bgonly)
        rows[c]["area"].append(bg.mean())

print("%-10s %8s %8s %8s" % ("cond", "unedit", "bg-only", "diff"))
for c in CONDS:
    w = np.mean(rows[c]["whole"]); b = np.mean(rows[c]["bgonly"])
    print("%-10s %8.2f %8.2f %8.2f" % (c, w, b, w - b))
print()
print("배경 면적 평균 %.3f, n=%d" % (np.mean(rows[CONDS[0]]["area"]), len(rows[CONDS[0]]["area"])))
a = np.array(rows[CONDS[0]]["area"])
for c in CONDS:
    w = np.array(rows[c]["whole"])
    print("%-10s corr(배경면적, unedit PSNR) = %+.3f" % (c, np.corrcoef(a, w)[0, 1]))
