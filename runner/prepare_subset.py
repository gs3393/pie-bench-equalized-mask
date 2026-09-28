"""PIE-Bench 700장에서 편집 유형별 층화 부분집합을 만든다.

출력: <out>/mapping_file.json + <out>/annotation_images (원본 심볼릭 링크)
선택은 카테고리 안에서 키를 정렬한 뒤 균등 간격으로 뽑아 결정적이다.
"""
import argparse, json, os, collections

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="PIE-bench 루트")
    ap.add_argument("--out", required=True)
    ap.add_argument("--total", type=int, default=150)
    args = ap.parse_args()

    mapping = json.load(open(os.path.join(args.src, "mapping_file.json")))
    by_cat = collections.defaultdict(list)
    for k, v in mapping.items():
        by_cat[v["editing_type_id"]].append(k)

    n_all = len(mapping)
    picked = {}
    counts = {}
    for cat in sorted(by_cat):
        keys = sorted(by_cat[cat])
        n = max(1, round(len(keys) * args.total / n_all))
        step = len(keys) / n
        idx = sorted({min(len(keys) - 1, int(i * step)) for i in range(n)})
        for i in idx:
            picked[keys[i]] = mapping[keys[i]]
        counts[cat] = len(idx)

    os.makedirs(args.out, exist_ok=True)
    link = os.path.join(args.out, "annotation_images")
    if not os.path.exists(link):
        os.symlink(os.path.abspath(os.path.join(args.src, "annotation_images")), link)
    with open(os.path.join(args.out, "mapping_file.json"), "w") as f:
        json.dump(picked, f, ensure_ascii=False)

    print("category counts:", counts)
    print("total picked:", len(picked))

if __name__ == "__main__":
    main()
