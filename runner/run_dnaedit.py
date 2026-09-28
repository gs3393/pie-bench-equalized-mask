"""DNAEdit (공식 xiechenxi99/DNAEdit_code, SD3 경로) 배치 실행 + 선택적 GT 마스크 블렌딩.

scripts/DNAEdit_utils.py 의 DNAEdit_SD3 를 복사하고 forward 루프의 latent 갱신 직후 한 줄을 넣었다:
  random_noise = random_noise*mask + ref*(1-mask)
ref 는 그 step 이 끝나는 노이즈 수준에 대응하는 "기록된 inversion latent" (last_lst[i+1-jmp]) 이고,
마지막 step 에서는 VAE 인코딩 원본 x_src 다. DirectEdit/FTEdit 조건과 같은 정의다.
하이퍼파라미터는 configs/DNAEdit_SD3_exp.yaml: T 40, T_start 13, src_cfg 1, tar_cfg 3.5, mvg 0.8, seed 0.
공식 스크립트는 DNA-Bench 의 long_mapping_file.json(긴 프롬프트)을 쓰지만 여기서는 다른 조건과 같은 PIE-Bench 원 프롬프트를 쓴다.
백본은 다른 조건과 같은 SD3.5-medium (논문 Table 1 의 DNAEdit-SD3.5 행에 해당).
"""
import argparse, os, sys, time, traceback, json
import numpy as np, torch
import PIL.Image as Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maskutil import prepare_latent_mask, load_mask, save_pair
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps


@torch.no_grad()
def dnaedit_sd3_masked(pipe, scheduler, x_src, src_prompt, tar_prompt, negative_prompt,
                       T_steps=40, src_guidance_scale=1.0, tar_guidance_scale=3.5, T_start=13, mvg=0.8,
                       latent_mask=None):
    device = x_src.device
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    pipe._num_timesteps = len(timesteps)

    pipe._guidance_scale = src_guidance_scale
    s_pe, s_npe, s_ppe, s_nppe = pipe.encode_prompt(prompt=src_prompt, prompt_2=None, prompt_3=None,
        negative_prompt=negative_prompt, do_classifier_free_guidance=pipe.do_classifier_free_guidance, device=device)
    pipe._guidance_scale = tar_guidance_scale
    t_pe, t_npe, t_ppe, t_nppe = pipe.encode_prompt(prompt=tar_prompt, prompt_2=None, prompt_3=None,
        negative_prompt=negative_prompt, do_classifier_free_guidance=pipe.do_classifier_free_guidance, device=device)

    pipe._guidance_scale = src_guidance_scale
    src_cfg = pipe.do_classifier_free_guidance
    if src_cfg:
        s_pe = torch.cat([s_npe, s_pe], dim=0); s_ppe = torch.cat([s_nppe, s_ppe], dim=0)
    pipe._guidance_scale = tar_guidance_scale
    tar_cfg = pipe.do_classifier_free_guidance
    if tar_cfg:
        t_pe = torch.cat([t_npe, t_pe], dim=0); t_ppe = torch.cat([t_nppe, t_ppe], dim=0)

    timesteps = torch.cat([timesteps, torch.tensor([0], device=timesteps.device)])
    inver_timesteps = torch.flip(timesteps, dims=[0])
    jmp = T_start
    last = x_src.clone()
    last_lst, v_lst, dx_lst = [], [], []
    random_noise = torch.randn_like(x_src)
    # ---- inversion (DNA) ----
    for i, (t_curr, t_prev) in enumerate(zip(inver_timesteps[:-1], inver_timesteps[1:])):
        if len(inver_timesteps) - 1 - i == jmp:
            break
        t_curr, t_prev = t_curr / 1000.0, t_prev / 1000.0
        x_curr = last
        x_prev = (t_prev - t_curr) / (1 - t_curr) * (random_noise - last) + last
        lmi = torch.cat([x_prev, x_prev], dim=0) if src_cfg else x_prev
        noise_pred_src = pipe.transformer(hidden_states=lmi, timestep=(t_prev * 1000).expand(lmi.shape[0]),
            encoder_hidden_states=s_pe, pooled_projections=s_ppe, joint_attention_kwargs=None, return_dict=False)[0]
        if src_cfg:
            u, c = noise_pred_src.chunk(2)
            noise_pred_src = u + src_guidance_scale * (c - u)
        delta_v = ((x_prev - x_curr) / (t_prev - t_curr) - noise_pred_src)
        x_prev = x_prev.to(torch.float32)
        last = x_prev - delta_v * (t_prev - t_curr)
        dx = delta_v * (t_prev - t_curr)
        last = last.to(delta_v.dtype)
        random_noise = random_noise.to(torch.float32)
        random_noise -= delta_v * (1 - t_curr)
        random_noise = random_noise.to(delta_v.dtype)
        last_lst.append(last); dx_lst.append(dx); v_lst.append(noise_pred_src)
    last_lst = last_lst[::-1]; v_lst = v_lst[::-1]; dx_lst = dx_lst[::-1]
    random_noise = last_lst[0]
    x_ref = x_src.clone()
    pipe._guidance_scale = tar_guidance_scale
    # ---- forward (edit) ----
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        if i < jmp:
            continue
        timestep = t_curr
        t_curr, t_prev = t_curr / 1000.0, t_prev / 1000.0
        zin = random_noise + dx_lst[i - jmp]
        lmi = torch.cat([zin, zin], dim=0) if tar_cfg else zin
        noise_pred_tgt = pipe.transformer(hidden_states=lmi, timestep=timestep.expand(lmi.shape[0]),
            encoder_hidden_states=t_pe, pooled_projections=t_ppe, joint_attention_kwargs=None, return_dict=False)[0]
        if tar_cfg:
            u, c = noise_pred_tgt.chunk(2)
            noise_pred_tgt = u + tar_guidance_scale * (c - u)
        x_ref = x_ref.to(torch.float32)
        x_ref = x_ref + (t_prev - t_curr) * (noise_pred_tgt - v_lst[i - jmp])
        x_ref = x_ref.to(noise_pred_tgt.dtype)
        v = noise_pred_tgt * mvg + (random_noise - x_ref) / t_curr * (1 - mvg)
        random_noise = random_noise.to(torch.float32)
        random_noise += v * (t_prev - t_curr)
        if latent_mask is not None:
            k = i + 1 - jmp
            ref = last_lst[k] if k < len(last_lst) else x_src
            m = latent_mask.to(random_noise.dtype)
            random_noise = random_noise * m + ref.to(random_noise.dtype) * (1 - m)
        random_noise = random_noise.to(v.dtype)
    return random_noise


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
    p.add_argument("--T_steps", type=int, default=40)
    p.add_argument("--T_start", type=int, default=13)
    p.add_argument("--src_cfg", type=float, default=1.0)
    p.add_argument("--tar_cfg", type=float, default=3.5)
    p.add_argument("--mvg", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

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
            torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
            image = Image.open(img_f).convert("RGB")
            src_prompt = it["original_prompt"].replace("[", "").replace("]", "")
            tar_prompt = it["editing_prompt"].replace("[", "").replace("]", "")
            x_src = encode(pipe, image, a.device)
            lm = None
            if use_mask:
                mimg = load_mask(a.mask_dir, stem, image.size)
                lm = prepare_latent_mask(mimg, (x_src.shape[-2], x_src.shape[-1]), a.device, torch.float32)
            if done == 0 and skipped == 0:
                print("mask:", None if lm is None else tuple(lm.shape), flush=True)
            x_tar = dnaedit_sd3_masked(pipe, scheduler, x_src, src_prompt, tar_prompt, "",
                                       a.T_steps, a.src_cfg, a.tar_cfg, a.T_start, a.mvg, latent_mask=lm)
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
