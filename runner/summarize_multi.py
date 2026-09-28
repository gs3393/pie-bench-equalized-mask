"""방법 × 마스크 조건 표와 순위 비교 (카드 251901459).

results/<method>_<cond>.csv (evaluate.py 출력) 를 읽어
  1. 방법 × 조건 × 7지표 평균표 (논문 표와 같은 스케일)
  2. 조건별로 5개 방법의 지표별 순위와 평균 순위 (논문 Table 1 의 순위 방식: 지표별 순위의 평균)
  3. 논문 Table 1 SD3.5 5행을 같은 방식으로 매긴 기준 순위
  4. 방법별 "마스크 없음 → GT 마스크" 가 편집 안 함 기준까지 닫은 비율
을 출력한다. 없는 파일은 건너뛴다.
"""
import argparse, os
import pandas as pd, numpy as np

METRICS = ["structure_distance", "psnr_unedit_part", "lpips_unedit_part", "mse_unedit_part",
           "ssim_unedit_part", "clip_similarity_target_image", "clip_similarity_target_image_edit_part"]
SHORT = ["Struct", "PSNR", "LPIPS", "MSE", "SSIM", "CLIPw", "CLIPe"]
LOWER = [True, False, True, True, False, False, False]

PAPER_SD35 = {  # Table 1, SD3.5 rows (Structure, PSNR, LPIPS, MSE, SSIM, CLIP whole, CLIP edit)
    "ftedit":    [21.06, 23.49, 90.25, 61.78, 86.23, 25.21, 21.78],
    "flowedit":  [23.13, 23.29, 92.81, 69.09, 85.22, 26.71, 23.59],
    "flowalign": [33.49, 24.06, 68.91, 54.29, 86.33, 25.64, 22.40],
    "dnaedit":   [11.03, 27.71, 60.51, 26.28, 90.13, 25.20, 22.25],
    "directedit":[14.65, 31.82, 31.36, 21.64, 92.28, 25.64, 22.67],
}
PAPER_SD35_NOMASK_DIRECTEDIT = [15.23, 26.18, 67.28, 34.00, 88.75, 26.13, 22.50]


def load(results, method, cond):
    # DirectEdit 의 초기 실행은 nomask.csv / gtmask.csv 로 저장되어 있다
    cands = [os.path.join(results, "%s_%s.csv" % (method, cond))]
    if method == "directedit":
        cands.append(os.path.join(results, "%s.csv" % cond))
    for p in cands:
        if os.path.exists(p):
            df = pd.read_csv(p)
            tag = [c.split("|")[0] for c in df.columns if "|" in c][0]
            return {m: pd.to_numeric(df["%s|%s" % (tag, m)], errors="coerce").mean() for m in METRICS}
    return None


def ranks(table):
    """table: {method: [7 values]} -> {method: [7 ranks]}, 평균 순위. 동점은 순위 평균(감사 2026-09-24 지적으로 수정)."""
    methods = list(table)
    R = {m: [] for m in methods}
    for j in range(7):
        vals = np.array([table[m][j] for m in methods], dtype=float)
        vals = vals if LOWER[j] else -vals
        vals = np.where(np.isinf(vals), np.sign(vals) * 1e30, vals)
        rk = pd.Series(vals).rank(method="average").values
        for i, m in enumerate(methods):
            R[m].append(float(rk[i]))
    return {m: (R[m], sum(R[m]) / 7) for m in methods}


def print_table(title, table):
    print("== %s ==" % title)
    print("%-12s" % "method" + "".join("%8s" % s for s in SHORT))
    for m, v in table.items():
        print("%-12s" % m + "".join("%8.2f" % x for x in v))
    print()


def print_ranks(title, table):
    rk = ranks(table)
    print("== %s : 지표별 순위 / 평균 순위 ==" % title)
    for m, (r, avg) in sorted(rk.items(), key=lambda kv: kv[1][1]):
        print("%-12s %s  avg %.2f" % (m, " ".join("%.1f" % x for x in r), avg))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.expanduser(os.environ.get("PIE_BENCH_WORKSPACE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "results"))
    ap.add_argument("--methods", nargs="+", default=["ftedit", "flowedit", "flowalign", "dnaedit", "directedit"])
    ap.add_argument("--conds", nargs="+", default=["nomask", "gtmask", "pixpaste"])
    a = ap.parse_args()

    ref = load(a.results, "vaebound", "")
    if ref is None:
        p = os.path.join(a.results, "vaebound.csv")
        df = pd.read_csv(p); ref = {m: pd.to_numeric(df["vaebound|" + m], errors="coerce").mean() for m in METRICS}

    data = {}
    for c in a.conds:
        for m in a.methods:
            v = load(a.results, m, c)
            if v is not None:
                data[(m, c)] = v

    # 1. 방법 × 조건 표
    for c in a.conds:
        tab = {m: [data[(m, c)][k] for k in METRICS] for m in a.methods if (m, c) in data}
        if tab:
            print_table("condition = %s (n=%d methods)" % (c, len(tab)), tab)
    print_table("no-edit reference (VAE round-trip)", {"vaebound": [ref[k] for k in METRICS]})

    # 2. 조건별 순위
    for c in a.conds:
        tab = {m: [data[(m, c)][k] for k in METRICS] for m in a.methods if (m, c) in data}
        if len(tab) >= 2:
            print_ranks("condition = %s" % c, tab)

    # 3. 논문 기준 순위 (SD3.5 5행)
    print_table("paper Table 1, SD3.5 rows", PAPER_SD35)
    print_ranks("paper Table 1, SD3.5 rows", PAPER_SD35)

    # 4. 방법별 gap closure (nomask -> gtmask, 기준 = vaebound)
    print("== gap closed by GT mask (nomask -> gtmask -> no-edit reference), 배경 4열 + Structure ==")
    print("%-12s" % "method" + "".join("%8s" % s for s in SHORT[:5]))
    for m in a.methods:
        if (m, "nomask") in data and (m, "gtmask") in data:
            row = []
            for k in METRICS[:5]:
                a0, a1, r = data[(m, "nomask")][k], data[(m, "gtmask")][k], ref[k]
                row.append((a1 - a0) / (r - a0) * 100 if abs(r - a0) > 1e-9 else float("nan"))
            print("%-12s" % m + "".join("%7.0f%%" % x for x in row))
    print()
    # CLIP-edit cost of masking
    print("== CLIP-edit: nomask -> gtmask (편집성 대가) ==")
    for m in a.methods:
        if (m, "nomask") in data and (m, "gtmask") in data:
            k = METRICS[6]
            print("%-12s %6.2f -> %6.2f  (%+.2f)" % (m, data[(m, "nomask")][k], data[(m, "gtmask")][k], data[(m, "gtmask")][k] - data[(m, "nomask")][k]))


if __name__ == "__main__":
    main()
