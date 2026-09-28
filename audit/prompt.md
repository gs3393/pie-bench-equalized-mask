# Prompt sent to gpt-6-astra (xhigh, read-only, cwd = audit-251901459)

You are an experiment integrity auditor AND a hostile referee. Start from the assumption that the evaluation is compromised somewhere and that the written claims overreach — your job is to find where. Be adversarial. Trust nothing the author tells you — verify everything yourself by reading the files. Do not rewrite the documents; report findings.

The author (a different model family from you) ran a controlled comparison of five text-guided image-editing methods on a 151-image subset of PIE-Bench with SD3.5-medium, under three mask conditions, and wrote (1) a published note and (2) two measurement reports. Everything is in the current working directory:

claims/ — published-note-editing-benchmark-background-metric.md, 2026-09-21-mask-ablation-results.md, 2026-09-24-equalized-mask-comparison.md (main target)
results/ — <method>_<cond>.csv (ftedit/flowedit/flowalign/dnaedit/directedit × nomask/gtmask/pixpaste), nomask/gtmask/automask.csv (DirectEdit earlier run), vaebound.csv, flowalign1024_nomask.csv, rescheck_flowalign_1024_vs_512.txt, summary_multi.txt, paired_stats.txt, PIE-bench-150_mapping_file.json, stat_*.csv
runner/ — author's scripts
src/ — unmodified upstream code (directedit__*, ftedit__*, flowedit__*, flowalign__*, dnaedit__*) + paper_directedit_2605.02417.txt
logs/ — generation and evaluation logs

Part 1 — Integrity checklist A–F (GT provenance; score normalization incl. bootstrap/rank-tie math; result existence & number fidelity incl. recomputing rankings and n=120/151 and done=151 failed=0; dead code / silent divergence of copied loops from upstream with every divergence listed against Appendix B; scope language; evaluation-type classification per condition incl. what pixpaste can and cannot show).

Part 2 — Claims audit, each VERIFIED / OVERSTATED / WRONG / CANNOT VERIFY with evidence and required wording:
1. 1.4 dB band / top four within 0.3 dB.
2. Remaining differences statistically resolvable but an order of magnitude smaller and traded against CLIP.
3. Ranking table and the "mask asymmetry produced the paper's #1" sentence.
4. What alignment buys vs FTEdit (LPIPS −0.34, MSE −0.20, CLIP-edit +0.62) and vs others; fairness of FTEdit as "ancestor minus alignment" given runner differences.
5. FlowAlign non-reproduction and rejection of the resolution hypothesis; validity of the 20-image 1024² test; possibility that the FlowAlign runner is wrong.
6. Masking costs ≤ ~0.3 CLIP-edit; background columns close 83–99%.
7. Appendix A z_ref definitions and index arithmetic (DNAEdit last_lst/jmp, FTEdit all_latents[-2-i]); fairness of per-step pasting vs FlowEdit's n_max=33 of 50.
8. Whether the five-method data strengthens/weakens/contradicts the published note.
9. Anything else.

Output: Part 1 blocks A–F; Part 2 blocks 1–9; MUST FIX; SHOULD FIX; Overall verdict for integrity and for claims. A false VERIFIED is worse than a CANNOT VERIFY.

(The exact prompt text as sent is recorded in the session transcript; this file is a faithful condensation of its structure and every listed item.)
