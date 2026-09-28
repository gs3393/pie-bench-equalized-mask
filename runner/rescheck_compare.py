"""FlowAlign 확인 실험 비교: 같은 이미지 집합에서 여러 실행의 지표 평균을 나란히 놓는다.
사용: rescheck_compare.py <base_run> <other_run> [<other_run> ...]
base_run 의 file_id 집합으로 다른 실행을 필터한다(151장 실행에서 20장 골라내기 등)."""
import os, sys, numpy as np, pandas as pd
R = os.path.expanduser("~/workspace/pie-bench/results")
METRICS = ["structure_distance", "psnr_unedit_part", "lpips_unedit_part", "mse_unedit_part",
           "ssim_unedit_part", "clip_similarity_target_image", "clip_similarity_target_image_edit_part"]
SHORT = ["Struct", "PSNR", "LPIPS", "MSE", "SSIM", "CLIPw", "CLIPe"]
PAPER = [33.49, 24.06, 68.91, 54.29, 86.33, 25.64, 22.40]


def load(name):
    df = pd.read_csv(os.path.join(R, name + ".csv")); df["file_id"] = df["file_id"].astype(str).str.zfill(12)
    tag = [c.split("|")[0] for c in df.columns if "|" in c][0]
    out = df[["file_id"]].copy()
    for k in METRICS: out[k] = pd.to_numeric(df[tag + "|" + k], errors="coerce")
    return out.set_index("file_id")


names = sys.argv[1:]
base = load(names[0]); ids = base.index
runs = {n: load(n).loc[ids] for n in names}
print("n = %d images; background n = %d" % (len(ids), int(base["psnr_unedit_part"].notna().sum())))
print("%-10s" % "metric" + "".join("%22s" % n[:21] for n in names) + "%10s" % "paper")
for k, s in zip(METRICS, SHORT):
    print("%-10s" % s + "".join("%22.2f" % runs[n][k].mean() for n in names) + "%10.2f" % PAPER[SHORT.index(s)])
