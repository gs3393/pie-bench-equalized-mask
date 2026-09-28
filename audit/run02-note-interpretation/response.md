# Reviewer response — run02, gpt-6-astra (xhigh), threadId 01a0e572-e976-7631-87fd-4c503422a14d

Verdict: publishable after MUST FIXes; SHOULD FIXes strongly recommended. Central result (background-spread compression, ranking sensitivity) supported and consistent with the first note. Not publishable as-is: save-path prose asserts pixel identity / CLIP invariance the data contradict; figure narratives assert mechanisms the comparisons do not isolate; compression summary misidentifies the least-compressed metrics; public repo has a stale analysis script.

## MUST FIX
1. "shifts every pixel while leaving the edit itself untouched" / "CLIP scores do not move, because the edit did not" / "pixel-for-pixel the same" — saved pixels differ inside the edit region too; mean CLIP-whole −0.060, CLIP-edit −0.087, per-image −2.78..+3.27. State: generation unchanged, saved pixels changed; give the CLIP means.
2. "account for most of the gap" — no defined aggregate gap across units (LPIPS ~34%, CLIP-whole ~6%, Structure overshoots). State sensitivity to saving in my runs; paper's path unverified.
3. Figure 1: the two panels are separate generation runs with/without per-step blending, not one edit clipped by a mask; masked panel keeps substantial cloud and restores the toy's legs/shadow; CLIP-edit already excludes out-of-mask pixels in both conditions, so out-of-mask cloud removal cannot directly explain the drop. Describe, don't explain.
4. Figure 2: unmasked run also distorts the figurine; masked run changes the cucumber's appearance (lengthwise section) with a straight edge near the boundary and a visible reflection; not "same cucumber minus shadow". Describe, don't explain.
5. "two columns that compress least are the non-background ones" — three non-background columns; spread reductions CLIP-edit 3.7%, CLIP-whole 18.9%, Structure 41.7%; Structure is not among the two least compressed; "largest differences" across unlike scales has no meaning.
6. Definitions: Structure Distance = MSE between whole-image DINO self-similarity matrices; CLIP-edit = target prompt vs edited image with out-of-annotation pixels zeroed (not a crop) — can see in-mask boundary artifacts, lacks the true surround; background metrics zero the edit region in both images and keep the whole-frame denominator. "CLIP-edit does not see the seam" too categorical.
14. Public repo: `rescheck_compare.py` is the obsolete version (hardcodes the mis-scored 1024 run) so `flowalign_checks{2,3}.sh` cannot regenerate rescheck2/3; docstrings stale (FlowEdit "one line", vae_roundtrip "bound"); README "Everything that was run" overclaims. Publish the corrected script, fix docstrings, describe what is reproducible.

## SHOULD FIX
7. Ranking prose: pixel compositing changes the images (Structure, CLIP-whole too), not just the background columns; two "fourth" results come from different comparisons.
8. Alignment: only PSNR is in dB; FlowEdit comparison has the SAME direction (higher PSNR, lower CLIP), not "the other way round"; FlowAlign PSNR interval includes zero; the FLUX ablation illustrates the design, does not estimate these differences.
9. "four of five rows landed near the paper" — DNAEdit Structure +3.22/LPIPS +10.88/MSE +6.77, FlowEdit MSE +5.66; say agreement varied by method/metric.
10. "unedited image wins outright at 0.55" — 0.55 is the 151-mean of the VAE reconstruction; per image not always lowest (611000000004: 18.55 vs masked FlowAlign 17.35).
11. Opening "to learn anything about the algorithm" broader than the first note (a within-system ablation can inform); equal masks are necessary for between-method comparison, not sufficient for isolation.
12. Pixel paste "floor" → constructed control; minima/maxima for background metrics only.
13. "re-saved the same 151 edits" — the check re-ran generation with the same seeds and a different output conversion; runner's round() vs torchvision's add_(0.5) → byte equivalence untested; torchvision default is normalize=False, FlowAlign passes True.
15. License text: MIT covers original contributions; no license found for FTEdit/DNAEdit portions — README does not establish permission; figures do contain images and prompt text.
16. Report §5.1 "해상도는 원인이 아니다" / "표본 크기 효과" too strong: 20 images do not exclude resolution effects in the authors' setup; 20→151 changes composition, not only size; report also retains the unchanged-CLIP assertion.
17. "a few tenths separate most pairs, 1.4 dB separates one" — four comparisons involving FlowEdit differ 1.14–1.43 dB; six pairs among the top four ≤0.29.

## NIT
18. Figure 2 "zoom" panel is a full-frame duplicate (forced mask border → full crop). Remove or crop manually.

## Confirmed present (rounds A–C)
five methods/six rows; FlowAlign-vs-FlowEdit confined to Structure; "six of seven" gone; sign-change values; rank attribution; ~3.5 GPU-h (label as excluding auxiliary checks/eval); Structure whole-image + compression acknowledged; shared spatial preprocessing vs identical intervention; exact restoration where processed mask = 0; reference path-specific, not a bound; scales, 120/151, exploratory intervals, means vs individuals; no alignment attribution; discard-background advice withdrawn; LPIPS ~1/3; resolution check 20 images; first mis-scoring disclosed; round-C ranking observation correct (DNAEdit 1.71, DirectEdit 2.57, FlowAlign 2.86, FlowEdit 3.71, FTEdit 4.14; fixed-range FlowAlign 1.43); figure numbers 21.1228→14.9896, 7.8978→12.1956 (PSNR 38.3284), 32.7005→17.6059; audit vs validation separated; public repo contents as advertised.
