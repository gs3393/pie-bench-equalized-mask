"""GT 마스크 조건에서 남는 방법 간 차이가 표본 잡음 안인지: 이미지별 짝지은 차이와 bootstrap 95% CI.
또 픽셀 합성 조건처럼 동점이 생기는 열은 평균 순위(동점은 순위 평균)로 다시 매긴다."""
import os, sys, numpy as np, pandas as pd
R = os.path.expanduser("~/workspace/pie-bench/results")
METRICS = ["structure_distance", "psnr_unedit_part", "lpips_unedit_part", "mse_unedit_part",
           "ssim_unedit_part", "clip_similarity_target_image", "clip_similarity_target_image_edit_part"]
SHORT = ["Struct", "PSNR", "LPIPS", "MSE", "SSIM", "CLIPw", "CLIPe"]
LOWER = [True, False, True, True, False, False, False]
METHODS = ["ftedit", "flowedit", "flowalign", "dnaedit", "directedit"]

def load(m, c):
    p = os.path.join(R, "%s_%s.csv" % (m, c))
    if not os.path.exists(p) and m == "directedit": p = os.path.join(R, "%s.csv" % c)
    df = pd.read_csv(p); df["file_id"] = df["file_id"].astype(str).str.zfill(12)
    tag = [x.split("|")[0] for x in df.columns if "|" in x][0]
    out = df[["file_id"]].copy()
    for k in METRICS: out[k] = pd.to_numeric(df[tag + "|" + k], errors="coerce")
    return out.set_index("file_id")

rng = np.random.default_rng(0)
def boot_ci(d, n=5000):
    d = d[~np.isnan(d)]; bs = [rng.choice(d, len(d), replace=True).mean() for _ in range(n)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(d)

for cond in ["gtmask", "nomask"]:
    print("== %s: DirectEdit minus other, per-image paired mean [95%% bootstrap CI] ==" % cond)
    D = {m: load(m, cond) for m in METHODS}
    for k, s, lb in zip(METRICS, SHORT, LOWER):
        row = []
        for m in METHODS:
            if m == "directedit": continue
            d = (D["directedit"][k] - D[m][k]).values
            mu, lo, hi, n = boot_ci(d)
            sig = "*" if (lo > 0 or hi < 0) else " "
            better = (mu < 0) if lb else (mu > 0)
            row.append("%-9s %+6.2f [%+6.2f,%+6.2f]%s%s" % (m, mu, lo, hi, sig, "D" if better else " "))
        print("  %-6s " % s + " | ".join(row))
    print("  (* = CI excludes 0; D = DirectEdit better on that metric)")
    print()

def avg_rank_ties(table):
    ms = list(table); out = {m: [] for m in ms}
    for j in range(7):
        vals = np.array([table[m][j] for m in ms], dtype=float)
        vals = vals if LOWER[j] else -vals
        vals = np.where(np.isinf(vals), np.sign(vals) * 1e30, vals)
        rk = pd.Series(vals).rank(method="average").values
        for i, m in enumerate(ms): out[m].append(rk[i])
    return {m: (out[m], float(np.mean(out[m]))) for m in ms}

for cond in ["nomask", "gtmask", "pixpaste"]:
    tab = {m: [load(m, cond)[k].mean() for k in METRICS] for m in METHODS}
    print("== %s: 동점 평균 순위 (7열) / 배경 4열 제외 3열 ==" % cond)
    rk = avg_rank_ties(tab)
    for m, (r, a) in sorted(rk.items(), key=lambda kv: kv[1][1]):
        r3 = np.mean([r[0], r[5], r[6]])
        print("  %-11s %s  avg7 %.2f | avg3(Struct,CLIPw,CLIPe) %.2f" % (m, " ".join("%.1f" % x for x in r), a, r3))
    print()
