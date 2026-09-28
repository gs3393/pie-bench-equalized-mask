"""DirectEdit SD3.5 배치 실행 (마스크 조건만 바꿔 돌리는 판).

DirectEdit/scripts/edit_real_sd35.py 를 그대로 옮기고 두 가지만 바꿨다.
  1. 마스크 디렉터리를 --mask_dir 로 받는다. 원본은 <src_path>/mask_generated 고정.
     --mask_dir none 이면 마스크 없이 돈다.
  2. 이미 저장된 결과는 건너뛴다 (중단 후 재개용).
나머지 하이퍼파라미터·저장 형식은 원본과 같다.
"""
import argparse, os, traceback, time
import torch
import PIL.Image as Image

from mmdit.sd35_pipeline import StableDiffusion3Pipeline
from inversion.flow_direct_correction_inv_sd35 import Accurate_Inversion_SD3
from inversion.inv_utils import fix_seed, load_PIE_images, view_images
from controller import attn_norm_ctrl_sd35


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--num_steps", type=int, default=30)
    p.add_argument("--skip_steps", type=int, default=0)
    p.add_argument("--inv_cfg", type=float, default=1.0)
    p.add_argument("--recov_cfg", type=float, default=2.0)
    p.add_argument("--ly_ratio", type=float, default=0.0)
    p.add_argument("--attn_ratio", type=float, default=0.3)
    p.add_argument("--src_path", type=str, required=True)
    p.add_argument("--saved_path", type=str, required=True)
    p.add_argument("--mask_dir", type=str, default="none")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = get_parser()
    fix_seed(args.seed)
    torch.Generator(device=args.device).manual_seed(args.seed)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
    pipe = pipe.to(args.device)
    pipe.transformer.eval()
    pipe.vae.eval()

    invf = Accurate_Inversion_SD3(pipe, args.num_steps, args.device, args.inv_cfg,
                                  args.recov_cfg, args.skip_steps, args.saved_path)

    ori, edi, imgs, _, _, _ = load_PIE_images(
        args.src_path, edit_category_list=[str(i) for i in range(10)])
    total = len(imgs) if args.limit <= 0 else min(args.limit, len(imgs))
    os.makedirs(args.saved_path, exist_ok=True)
    use_mask = args.mask_dir.lower() != "none"
    print("images: %d | mask_dir: %s" % (total, args.mask_dir))

    t0 = time.time()
    done = skipped = failed = 0
    for i in range(total):
        img_f = imgs[i]
        stem = os.path.splitext(os.path.basename(img_f))[0]
        out_stem = os.path.join(args.saved_path, stem)
        if os.path.exists(out_stem + "_edited.png"):
            skipped += 1
            continue
        try:
            image = Image.open(img_f).convert("RGB")
            mask = None
            if use_mask:
                mp = os.path.join(args.mask_dir, stem + "_mask.png")
                if os.path.exists(mp):
                    mask = Image.open(mp).convert("L")
                    if mask.size != image.size:
                        mask = mask.resize(image.size, Image.NEAREST)
                else:
                    raise FileNotFoundError("mask missing: " + mp)

            src_prompt = ori[i].replace("[", "").replace("]", "")
            tar_prompt = edi[i].replace("[", "").replace("]", "")
            prompts = [src_prompt, tar_prompt]

            attn_norm_ctrl_sd35.register_attention_control_sd35(pipe, None, None)
            all_latents, delta_list = invf.euler_flow_inversion(prompt=src_prompt, image=img_f)

            ctrl_ada = attn_norm_ctrl_sd35.Adalayernorm_replace(
                prompts, args.num_steps, args.ly_ratio, pipe.tokenizer, pipe.tokenizer_3, device=args.device)
            ctrl_attn = attn_norm_ctrl_sd35.SD3attentionreplace(prompts, args.num_steps, args.attn_ratio)
            attn_norm_ctrl_sd35.register_attention_control_sd35(pipe, ctrl_attn, ctrl_ada)

            image_list = invf.direct_inversion(
                prompts, controller=ctrl_ada, all_latents=all_latents, delta_list=delta_list,
                original_size=image.size, mask_image=mask)
            view_images(image_list, out_stem)
            done += 1
            if done % 10 == 0:
                el = time.time() - t0
                print("[%d/%d] %.1fs elapsed, %.1fs/img" % (i + 1, total, el, el / max(done, 1)), flush=True)
        except Exception as e:
            failed += 1
            print("ERROR on %s: %s" % (img_f, e))
            traceback.print_exc()
            continue

    print("done=%d skipped=%d failed=%d total_time=%.1fs" % (done, skipped, failed, time.time() - t0))
