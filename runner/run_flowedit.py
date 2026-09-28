"""FlowEdit (공식 fallenshock 저장소, SD3 경로) 배치 실행 + 선택적 GT 마스크 블렌딩.

FlowEdit_utils.FlowEditSD3 를 복사하고 한 줄만 넣었다:
  zt_edit = zt_edit*mask + x_src*(1-mask)   # 매 step, ODE 갱신 직후
FlowEdit 의 운반 변수 zt_edit 는 x_src + 누적 델타 (깨끗한 좌표) 이므로, 붙여 넣는 "원본"은 매 step 같은 x_src 다.
마지막 step 에서 마스크 밖은 VAE 인코딩 원본 그 자체가 되어 DirectEdit/FTEdit 조건과 같은 정의다.
inversion 이 없으니 "기록된 inversion latent" 대신 x_src 를 쓴다는 것이 이 방법의 대응물이다.
하이퍼파라미터는 SD3_exp.yaml: T 50, n_avg 1, src_cfg 3.5, tar_cfg 13.5, n_min 0, n_max 33, seed 42.
백본은 다른 조건과 같은 SD3.5-medium (논문 Table 1 의 FlowEdit-SD3.5 행에 해당).
"""
import argparse, os, sys, time, traceback
import numpy as np, torch
import PIL.Image as Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maskutil import prepare_latent_mask, load_mask, save_pair
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from FlowEdit_utils import calc_v_sd3


@torch.no_grad()
def flowedit_sd3_masked(pipe, scheduler, x_src, src_prompt, tar_prompt, negative_prompt,
                        T_steps=50, n_avg=1, src_guidance_scale=3.5, tar_guidance_scale=13.5,
                        n_min=0, n_max=33, latent_mask=None):
    device = x_src.device
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    pipe._num_timesteps = len(timesteps)
    pipe._guidance_scale = src_guidance_scale
    s_pe, s_npe, s_ppe, s_nppe = pipe.encode_prompt(prompt=src_prompt, prompt_2=None, prompt_3=None,
        negative_prompt=negative_prompt, do_classifier_free_guidance=pipe.do_classifier_free_guidance, device=device)
    pipe._guidance_scale = tar_guidance_scale
    t_pe, t_npe, t_ppe, t_nppe = pipe.encode_prompt(prompt=tar_prompt, prompt_2=None, prompt_3=None,
        negative_prompt=negative_prompt, do_classifier_free_guidance=pipe.do_classifier_free_guidance, device=device)
    st_pe = torch.cat([s_npe, s_pe, t_npe, t_pe], dim=0)
    st_ppe = torch.cat([s_nppe, s_ppe, t_nppe, t_ppe], dim=0)
    zt_edit = x_src.clone()
    assert n_min == 0, "n_min>0 branch not ported"
    for i, t in enumerate(timesteps):
        if T_steps - i > n_max:
            continue
        t_i = t / 1000
        t_im1 = (timesteps[i + 1]) / 1000 if i + 1 < len(timesteps) else torch.zeros_like(t_i).to(t_i.device)
        V_delta_avg = torch.zeros_like(x_src)
        for k in range(n_avg):
            fwd_noise = torch.randn_like(x_src).to(x_src.device)
            zt_src = (1 - t_i) * x_src + t_i * fwd_noise
            zt_tar = zt_edit + zt_src - x_src
            lmi = torch.cat([zt_src, zt_src, zt_tar, zt_tar])
            Vt_src, Vt_tar = calc_v_sd3(pipe, lmi, st_pe, st_ppe, src_guidance_scale, tar_guidance_scale, t)
            V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src)
        zt_edit = zt_edit.to(torch.float32)
        zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
        if latent_mask is not None:
            m = latent_mask.to(zt_edit.dtype)
            zt_edit = zt_edit * m + x_src.to(zt_edit.dtype) * (1 - m)
        zt_edit = zt_edit.to(V_delta_avg.dtype)
    return zt_edit


def encode(pipe, image, device):
    image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
    x = pipe.image_processor.preprocess(image).to(device).half()
    with torch.autocast("cuda"), torch.inference_mode():
        z = pipe.vae.encode(x).latent_dist.mode()
    return (z - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor


def decode(pipe, z):
    zd = (z / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with torch.autocast("cuda"), torch.inference_mode():
        img = pipe.vae.decode(zd, return_dict=False)[0]
    return pipe.image_processor.postprocess(img)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_path", required=True)
    p.add_argument("--saved_path", required=True)
    p.add_argument("--mask_dir", default="none")
    p.add_argument("--T_steps", type=int, default=50)
    p.add_argument("--n_avg", type=int, default=1)
    p.add_argument("--src_cfg", type=float, default=3.5)
    p.add_argument("--tar_cfg", type=float, default=13.5)
    p.add_argument("--n_min", type=int, default=0)
    p.add_argument("--n_max", type=int, default=33)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    import json
    mapping = json.load(open(os.path.join(a.src_path, "mapping_file.json")))
    items = list(mapping.values())
    total = len(items) if a.limit <= 0 else min(a.limit, len(items))
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16).to(a.device)
    scheduler = pipe.scheduler
    os.makedirs(a.saved_path, exist_ok=True)
    use_mask = a.mask_dir.lower() != "none"
    print("images: %d | mask_dir: %s" % (total, a.mask_dir), flush=True)
    t0 = time.time(); done = skipped = failed = 0
    for i in range(total):
        it = items[i]
        img_f = os.path.join(a.src_path, "annotation_images", it["image_path"])
        stem = os.path.splitext(os.path.basename(img_f))[0]
        out = os.path.join(a.saved_path, stem)
        if os.path.exists(out + "_edited.png"):
            skipped += 1; continue
        try:
            torch.manual_seed(a.seed)
            image = Image.open(img_f).convert("RGB")
            src_prompt = it["original_prompt"].replace("[", "").replace("]", "")
            tar_prompt = it["editing_prompt"].replace("[", "").replace("]", "")
            x_src = encode(pipe, image, a.device)
            lm = None
            if use_mask:
                mimg = load_mask(a.mask_dir, stem, image.size)
                lm = prepare_latent_mask(mimg, (x_src.shape[-2], x_src.shape[-1]), a.device, torch.float32)
                if done == 0 and skipped == 0:
                    print("Mask enabled. latent mask", tuple(lm.shape), flush=True)
            x_tar = flowedit_sd3_masked(pipe, scheduler, x_src, src_prompt, tar_prompt, "",
                                        a.T_steps, a.n_avg, a.src_cfg, a.tar_cfg, a.n_min, a.n_max, latent_mask=lm)
            save_pair([decode(pipe, x_src), decode(pipe, x_tar)], out)
            done += 1
            if done % 10 == 0:
                el = time.time() - t0
                print("[%d/%d] %.1fs elapsed, %.1fs/img" % (i + 1, total, el, el / done), flush=True)
        except Exception as e:
            failed += 1
            print("ERROR on %s: %s" % (img_f, e)); traceback.print_exc(); continue
    print("done=%d skipped=%d failed=%d total_time=%.1fs" % (done, skipped, failed, time.time() - t0))


if __name__ == "__main__":
    main()
