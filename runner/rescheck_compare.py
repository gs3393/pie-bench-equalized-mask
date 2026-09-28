"""FlowAlign 1024² (20장) vs 512² (같은 20장, 151장 실행에서 필터) 비교."""
import os, json, numpy as np, pandas as pd
R = os.path.expanduser("~/workspace/pie-bench/results")
METRICS = ["structure_distance", "psnr_unedit_part", "lpips_unedit_part", "mse_unedit_part",
           "ssim_unedit_part", "clip_similarity_target_image", "clip_similarity_target_image_edit_part"]
SHORT = ["Struct", "PSNR", "LPIPS", "MSE", "SSIM", "CLIPw", "CLIPe"]
def load(name):
    df = pd.read_csv(os.path.join(R, name + ".csv")); df["file_id"] = df["file_id"].astype(str).str.zfill(12)
    tag = [c.split("|")[0] for c in df.columns if "|" in c][0]
    out = df[["file_id"]].copy()
    for k in METRICS: out[k] = pd.to_numeric(df[tag + "|" + k], errors="coerce")
    return out.set_index("file_id")
a = load("flowalign1024_nomask"); b = load("flowalign_nomask").loc[a.index]
print("n =", len(a))
print("%-10s %8s %8s %8s" % ("metric", "1024", "512", "paper"))
paper = dict(zip(SHORT, [33.49, 24.06, 68.91, 54.29, 86.33, 25.64, 22.40]))
for k, s in zip(METRICS, SHORT):
    print("%-10s %8.2f %8.2f %8.2f" % (s, a[k].mean(), b[k].mean(), paper[s]))
