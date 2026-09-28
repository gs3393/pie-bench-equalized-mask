# Experiment Audit Report

**Date**: 2026-09-24
**Auditor**: External reviewer backend — Codex MCP, gpt-6-astra, reasoning xhigh, sandbox read-only (cross-model; executor was Claude)
**Project**: 카드 251901459 마스크 조건을 맞춘 다중 방법 비교 (+ 선행 카드 250790709 보고서, 공개 글)
**Bundle audited**: claims/ (3 문서), results/ (CSV 전부·요약·paired stats), runner/ (스크립트 전부), src/ (상류 원본 코드·논문 텍스트), logs/
**Trace**: `audit/prompt.md`, `audit/response.md`, `audit/meta.md` (이 폴더)

## Overall Verdict: FAIL (초판 기준) → 반영 후 재감사는 하지 않음

## Integrity Status: fail (초판) — 조작 정황 없음, 채점 오류·순위 동점·미공개 편차가 원인

## Checks

### A. Ground Truth Provenance: WARN
- 평가는 `annotation_images` 원본과 `mapping_file.json`의 GT 마스크를 읽는다. 모델 출력에서 유도한 기준을 자기 자신과 비교하는 곳은 없다. `evaluate.py`·`matrics_calculator.py`는 DirectEdit 공개본과 바이트 동일.
- `automask`는 GT bbox·카테고리를 받은 SAM2 마스크(주석 보조 프록시). `pixpaste`는 평가와 같은 마스크로 원본 픽셀을 되돌린 구성된 통제군. `vaebound`는 DirectEdit 경로의 무편집 기준이며 보편 상한이 아님.

### B. Score Normalization: FAIL
- 지표 정규화 문제 없음. gap closure 공식은 옳으나 평균 차이의 비율일 뿐 달성률·상한이 아님.
- `summarize_multi.py`가 동점을 사전 순으로 깸 → 픽셀 합성 순위 무효, 논문 5행 CLIP-whole 동점(25.64) 임의 처리. → **수정(동점 평균)**. 구버전 `summarize.py`의 이중 스케일 결함 → **삭제**.
- FlowAlign 저장 정규화가 상류(이미지별 min/max)와 다름(고정 범위) → 부록 B에 기재, 5.1절에서 시험.

### C. Result File Existence: FAIL (주표는 PASS)
- 15개 method×cond CSV 모두 존재, 151 ID, 배경 열 120 비결측(결측 31장 = 전체 마스크). 세 표 105셀·기준 행 7셀 모두 일치. 로그 done=151 failed=0.
- 논문 5행 순위 1.57/3.14 → 동점 평균 1.50/3.21로 **정정**.
- **1024² 해상도 실험이 편집본의 우하단 1/4만 채점** (`evaluate.py`의 우하단 512² crop이 2068×1024 캔버스에 적용) → 결과 철회, 저장 시 원본 크기로 되돌려 재실행.

### D. Dead Code Detection: FAIL ("블렌딩 한 줄만" 주장 기준)
- 마스크 삽입은 다섯 runner 모두 서술한 위치에서 작동. 인덱스 산술(FTEdit `-2-i`, DNAEdit `last_lst[i+1-jmp]`) 확인.
- 미공개 편차: FlowAlign 저장 정규화, DNAEdit 소스/타깃 CFG 플래그 분리(상류 상태가 비일관), FlowEdit·DNAEdit 이미지별 재시드, FlowEdit `n_min>0` 분기 제거, FTEdit 원 해상도 crop. → **부록 B에 전부 기재, "한 줄만" 철회**.
- FTEdit 컨트롤러 패치는 호환성 수리로 판단(불공정 근거 없음). 상류 미사용 코드(FlowAlign `n_start`·null 임베딩 등)는 결과에 무관.

### E. Scope Assessment: WARN
- 데이터 범위(151장·시드 1·SD3.5-M·GT 마스크)는 명시됨. 인과 주장 3건이 설계를 넘음: 마스크 비대칭이 논문 1위의 원인, FTEdit 비교 = 정렬 기여, 해상도 가설 기각. → **모두 철회·축소**.
- 선행 보고서(250790709)의 "변별력 상실"·"Structure만 유효"는 새 데이터와 모순 → **정정 블록 추가**.

### F. Evaluation Type
- nomask: real_gt 기준 평가 | gtmask: 주석 보조 개입 + real_gt | automask: 주석 보조 프록시 | vaebound: 무편집 기준 통제군 | pixpaste: 구성된 통제군(배경 완벽은 구성상 보장) | flowalign1024(초판): 공간 채점 오류로 무효

## Action Items
- [x] 1024² 실험 철회·재실행 (저장 시 원본 크기로 복원) — 결과는 보고서 5.1절
- [x] 정렬 기여 귀속 철회 → 구성된 시스템 비교로 서술
- [x] "마스크 비대칭이 논문 1위를 만들었다" → 이 실험 안의 감도 진술
- [x] 미공개 편차 4건 부록 B 기재
- [x] 순위 동점 평균으로 재계산(스크립트·표)
- [x] 83~102%, FlowEdit 예외, 압축 배율 비일률 반영
- [x] 통제군을 시스템 성능과 구분
- [x] 선행 보고서 정정 블록
- [x] 권고 8건(용어·CI 성격·n=17·PSNR 단위·7.7%·기준 경로·summarize.py 삭제·매니페스트 한계 명시)
- [ ] 재현 매니페스트(생성 이미지·마스크·버전·커밋) 보존 — 미수행. 원격에만 존재

## Claim Impact
- C1 배경 열 압축(PSNR 4.46→1.43 dB, 상위 넷 0.29 dB): **supported (needs qualifier — PSNR 단위, 다른 열은 각자 스케일)**
- C2 남는 차이 일부 유의·압축 배율: **supported with qualifier (배율 비일률, 탐색적 CI)**
- C3 순위 표·마스크 비대칭 인과: **ranking supported; causal claim unsupported → 철회**
- C4 정렬이 산 것: **unsupported as component effect → 시스템 비교로 재서술**
- C5 FlowAlign 재현 실패·해상도 기각: **non-reproduction supported as "adapted run differs"; resolution rejection unsupported → 철회, 재실험**
- C6 CLIP-edit 비용·gap closure: **needs qualifier (평균 −0.15~+0.29, 83~102%)**
- C7 부록 A 정의·인덱스: **supported (indices), equivalence claim overstated → 축소**
- C8 공개 글 중심 주장: **supported**
