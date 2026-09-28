"""FTEdit (SD3.5) 배치 실행 + 선택적 GT 마스크 블렌딩.

FTEdit/edit_real_sd35.py 의 흐름을 그대로 따르고 두 가지만 바꿨다.
  1. edit_img_with_residual 을 복사해 소스 분기 잔차 보정 직후에 한 줄을 넣는다:
       tar = tar*mask + src*(1-mask)   # src 는 보정 후 = 기록된 inversion latent
     FTEdit 은 소스 분기를 매 step 잔차로 원본 inversion 궤적에 맞추므로 (all_latents[-2-i] 로 덮는 것과 동치)
     붙여 넣는 값은 DirectEdit 과 같은 "기록된 inversion latent" 다. 마지막 step 에서는 VAE 인코딩 원본이다.
  2. diffusers 0.32.2 의 SD3.5 마지막 블록(norm1_context 에 emb 없음)에서 FTEdit 컨트롤러 등록이 깨질 수 있어,
     emb 가 없는 모듈의 forward 를 등록 뒤 원래대로 되돌리고 컨트롤러의 층 수를 맞춘다 (DirectEdit 이 넣은 guard 와 같은 효과).
하이퍼파라미터는 README 의 PIE 설정: inv_cfg 1, recov_cfg 2, skip 0, ly_ratio 1.0, attn_ratio 0.15, 30 step, fixpoint 2 / (0,5).
dtype 은 다른 조건과 맞추기 위해 fp16 (README 는 bf16).
"""
import argparse, os, sys, time, traceback
import numpy as np, torch
import PIL.Image as Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maskutil import prepare_latent_mask, load_mask, save_pair

from mmdit.sd35_pipeline import StableDiffusion3Pipeline, retrieve_timesteps
from inversion.flow_fixpoint_residual_new import Inversed_flow_fixpoint_residual
from inversion.inv_utils import fix_seed, load_PIE_images
from controller import attn_norm_ctrl_sd35


class FTEditMasked(Inversed_flow_fixpoint_residual):
    @torch.no_grad()
    def encode_latent(self, image_path):
        """FTEdit 원본은 load_1024 + bfloat16 고정이라 fp16 파이프라인과 맞지 않는다. 원 해상도(16 배수 crop)에서 VAE dtype 으로 인코딩한다."""
        img = Image.open(image_path).convert("RGB")
        img = img.crop((0, 0, img.width - img.width % 16, img.height - img.height % 16))
        x = torch.from_numpy(np.asarray(img)).float() / 127.5 - 1
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device).to(self.model.vae.dtype)
        z = self.model.vae.encode(x)["latent_dist"].mean
        return (z - self.model.vae.config.shift_factor) * self.model.vae.config.scaling_factor

    @torch.no_grad()
    def edit_masked(self, prompt, all_latents, controller, latent_mask=None):
        latent = torch.cat([all_latents[-1].clone().detach()] * 2, dim=0).to(self.device)
        s_pe, s_npe, s_ppe, s_nppe = self.get_embeddings(prompt[0])
        t_pe, t_npe, t_ppe, t_nppe = self.get_embeddings(prompt[1])
        prompt_embeds = torch.cat([s_pe, t_pe], dim=0)
        negative_prompt_embeds = torch.cat([s_npe, t_npe], dim=0)
        pooled = torch.cat([s_ppe, t_ppe], dim=0)
        npooled = torch.cat([s_nppe, t_nppe], dim=0)
        if self.recov_cfg > 0:
            all_pe = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            all_ppe = torch.cat([npooled, pooled], dim=0)
        else:
            all_pe, all_ppe = prompt_embeds, pooled
        timesteps, _ = retrieve_timesteps(self.model.scheduler, self.num_steps, self.device, None)
        self.model._num_timesteps = len(timesteps)
        for i in range(self.num_steps):
            if i < self.skip_steps:
                if controller is not None:
                    controller.cur_step += 1
                continue
            t = timesteps[i]
            lmi = torch.cat([latent] * 2) if self.recov_cfg > 0 else latent
            noise_pred = self.model.transformer(
                hidden_states=lmi, timestep=t.expand(lmi.shape[0]),
                encoder_hidden_states=all_pe, pooled_projections=all_ppe,
                joint_attention_kwargs=None, return_dict=False)[0]
            if self.recov_cfg > 0:
                u, c = noise_pred.chunk(2)
                noise_pred = u + self.recov_cfg * (c - u)
            sample = latent.to(torch.float32)
            sigma = self.model.scheduler.sigmas[i]
            sigma_next = self.model.scheduler.sigmas[i + 1]
            prev_sample = sample + (sigma_next - sigma) * noise_pred
            src_prev = prev_sample[0, :].clone().detach()
            residual = all_latents[-2 - (i - self.skip_steps)] - src_prev
            prev_sample[0, :] += residual.squeeze(0)
            if latent_mask is not None:
                m = latent_mask.to(prev_sample.dtype)
                prev_sample[1:2] = prev_sample[1:2] * m + prev_sample[0:1] * (1 - m)
            latent = prev_sample.to(noise_pred.dtype)
        img1 = self.latent2image(latent[0].unsqueeze(0))
        img2 = self.latent2image(latent[1].unsqueeze(0))
        self.model.maybe_free_model_hooks()
        return img1, img2


def register_safe(pipe, ctrl_attn, ctrl_ada):
    """emb 가 없는 norm1_context 는 등록 후 원래 forward 로 되돌린다."""
    restore = {}
    for name, blk in pipe.transformer.transformer_blocks.named_children():
        nc = getattr(blk, "norm1_context", None)
        if nc is not None and not hasattr(nc, "emb"):
            restore[name] = type(nc).forward.__get__(nc, type(nc))
    attn_norm_ctrl_sd35.register_attention_control_sd35(pipe, ctrl_attn, ctrl_ada)
    for name, blk in pipe.transformer.transformer_blocks.named_children():
        if name in restore:
            blk.norm1_context.forward = restore[name]
    if ctrl_ada is not None and restore:
        ctrl_ada.num_adanorm -= len(restore)
    return len(restore)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_path", required=True)
    p.add_argument("--saved_path", required=True)
    p.add_argument("--mask_dir", default="none")
    p.add_argument("--num_steps", type=int, default=30)
    p.add_argument("--skip_steps", type=int, default=0)
    p.add_argument("--inv_cfg", type=float, default=1.0)
    p.add_argument("--recov_cfg", type=float, default=2.0)
    p.add_argument("--ly_ratio", type=float, default=1.0)
    p.add_argument("--attn_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    fix_seed(a.seed)
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16).to(a.device)
    pipe.transformer.eval(); pipe.vae.eval()
    invf = FTEditMasked(pipe, a.num_steps, a.device, a.inv_cfg, a.recov_cfg, a.skip_steps, a.saved_path)
    ori, edi, imgs, _, _, _ = load_PIE_images(a.src_path, edit_category_list=[str(i) for i in range(10)])
    total = len(imgs) if a.limit <= 0 else min(a.limit, len(imgs))
    os.makedirs(a.saved_path, exist_ok=True)
    use_mask = a.mask_dir.lower() != "none"
    print("images: %d | mask_dir: %s" % (total, a.mask_dir), flush=True)
    t0 = time.time(); done = skipped = failed = 0
    for i in range(total):
        img_f = imgs[i]
        stem = os.path.splitext(os.path.basename(img_f))[0]
        out = os.path.join(a.saved_path, stem)
        if os.path.exists(out + "_edited.png"):
            skipped += 1; continue
        try:
            image = Image.open(img_f).convert("RGB")
            src_prompt = ori[i].replace("[", "").replace("]", "")
            tar_prompt = edi[i].replace("[", "").replace("]", "")
            prompts = [src_prompt, tar_prompt]
            n_restored = register_safe(pipe, None, None)
            all_latents = invf.euler_flow_inversion(prompt=src_prompt, image=img_f, num_fixpoint_steps=2, average_step_ranges=(0, 5))
            ctrl_ada = attn_norm_ctrl_sd35.Adalayernorm_replace(prompts, a.num_steps, float(a.ly_ratio), pipe.tokenizer, pipe.tokenizer_3, device=a.device)
            ctrl_attn = attn_norm_ctrl_sd35.SD3attentionreplace(prompts, a.num_steps, float(a.attn_ratio))
            register_safe(pipe, ctrl_attn, ctrl_ada)
            lm = None
            if use_mask:
                mimg = load_mask(a.mask_dir, stem, image.size)
                lat_hw = (all_latents[0].shape[-2], all_latents[0].shape[-1])
                lm = prepare_latent_mask(mimg, lat_hw, a.device, torch.float32)
            if done == 0 and skipped == 0:
                print("mask:", None if lm is None else tuple(lm.shape), "| restored norm modules:", n_restored, flush=True)
            img1, img2 = invf.edit_masked(prompts, all_latents, ctrl_ada, latent_mask=lm)
            save_pair([np.squeeze(img1), np.squeeze(img2)], out)
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
