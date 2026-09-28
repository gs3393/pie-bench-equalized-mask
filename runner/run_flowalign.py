"""FlowAlign (공식 FlowAlign/FlowAlign 저장소, SD3 경로) 배치 실행 + 선택적 GT 마스크 블렌딩.

diffusion/editing/sd3_edit.py 의 SD3FlowAlign.sample 을 복사하고 xt 갱신 직후 한 줄을 넣었다:
  xt = xt*mask + zsrc*(1-mask)
FlowAlign 의 운반 변수 xt 는 zsrc 에서 출발해 편집된 깨끗한 latent 로 끝나므로(FlowEdit 과 같은 좌표),
붙여 넣는 "원본"은 매 step 같은 zsrc 다. 마지막 step 에서 마스크 밖은 VAE 인코딩 원본 그 자체다.
하이퍼파라미터는 run_edit.py 기본값: NFE 33, cfg 13.5, n_start 0, seed 123. 해상도는 PIE-Bench 에 맞춰 512 (기본값 1024 대신).
원 저장소의 encode 는 latent_dist.sample() 이지만 다른 조건과 같이 mode() 를 쓴다 (원본 latent 를 확정적으로 만들기 위함).
그 외 편차: 이미지별 재시드, 저장 변환(--save_norm clamp 는 고정 범위, minmax 는 상류 save_image(normalize=True) 의 정책을 runner 가 직접 구현한 것이며 바이트 동일성은 시험하지 않음).
백본은 다른 조건과 같은 SD3.5-medium (논문 Table 1 의 FlowAlign-SD3.5 행에 해당).
"""
import argparse, os, sys, time, traceback, json
import numpy as np, torch
import PIL.Image as Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maskutil import prepare_latent_mask, load_mask, save_pair
from diffusion.editing.sd3_edit import SD3FlowAlign


class FlowAlignMasked(SD3FlowAlign):
    def encode(self, image):
        z = self.vae.encode(image).latent_dist.mode()
        return (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    @torch.no_grad()
    def sample_masked(self, src_img, src_prompt, tgt_prompt, null_prompt, NFE, img_shape, cfg_scale=13.5, latent_mask=None):
        src_prompt_emb = self.prepare_embed(src_prompt, None)
        tgt_prompt_emb = self.prepare_embed(tgt_prompt, None)
        self.scheduler.set_timesteps(NFE, device=self.transformer.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps
        zsrc = self.encode(src_img.to(self.vae.device).half())
        xt = zsrc.clone()
        for i, t in enumerate(timesteps):
            eps = torch.randn_like(zsrc)
            timestep = t.expand(1)
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1] if i + 1 < NFE else 0.0
            qt = (1 - sigma) * zsrc + sigma * eps
            pt = xt + qt - zsrc
            vpc = self.predict_vector(pt, timestep, tgt_prompt_emb[0], tgt_prompt_emb[1])
            vpn = self.predict_vector(pt, timestep, src_prompt_emb[0], src_prompt_emb[1])
            vp = vpn + cfg_scale * (vpc - vpn)
            vq = self.predict_vector(qt, timestep, src_prompt_emb[0], src_prompt_emb[1])
            xt = xt + (sigma_next - sigma) * (vp - vq) + 0.01 * (qt - sigma * vq - pt + sigma * vp)
            if latent_mask is not None:
                m = latent_mask.to(xt.dtype)
                xt = xt * m + zsrc.to(xt.dtype) * (1 - m)
        return zsrc, xt


def to_pil(img_tensor, mode="clamp"):
    """clamp: [-1,1] 고정 범위 매핑. minmax: 상류 run_edit.py 의 save_image(normalize=True) 와 같은 텐서별 min/max 정규화."""
    if mode == "minmax":
        lo, hi = float(img_tensor.min()), float(img_tensor.max())
        x = ((img_tensor - lo) / max(hi - lo, 1e-5) * 255).round().clamp(0, 255).to(torch.uint8)
    else:
        x = ((img_tensor.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)
    return Image.fromarray(x[0].permute(1, 2, 0).cpu().numpy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_path", required=True)
    p.add_argument("--saved_path", required=True)
    p.add_argument("--mask_dir", default="none")
    p.add_argument("--NFE", type=int, default=33)
    p.add_argument("--cfg_scale", type=float, default=13.5)
    p.add_argument("--img_shape", type=int, default=512)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_norm", default="clamp", choices=["clamp", "minmax"])
    a = p.parse_args()

    mapping = json.load(open(os.path.join(a.src_path, "mapping_file.json")))
    items = list(mapping.values())
    total = len(items) if a.limit <= 0 else min(a.limit, len(items))
    sampler = FlowAlignMasked(model_key="stabilityai/stable-diffusion-3.5-medium", device=a.device).to(device=a.device)
    tf = transforms.Compose([transforms.Resize((a.img_shape, a.img_shape)), transforms.ToTensor()])
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
            pil = Image.open(img_f).convert("RGB")
            src_img = tf(pil).unsqueeze(0) * 2.0 - 1.0
            src_prompt = it["original_prompt"].replace("[", "").replace("]", "")
            tar_prompt = it["editing_prompt"].replace("[", "").replace("]", "")
            lm = None
            if use_mask:
                mimg = load_mask(a.mask_dir, stem, (a.img_shape, a.img_shape))
                lm = prepare_latent_mask(mimg, (a.img_shape // 8, a.img_shape // 8), a.device, torch.float32)
            if done == 0 and skipped == 0:
                print("mask:", None if lm is None else tuple(lm.shape), flush=True)
            zsrc, xt = sampler.sample_masked(src_img, src_prompt, tar_prompt, "", a.NFE, (a.img_shape, a.img_shape), a.cfg_scale, latent_mask=lm)
            with torch.no_grad():
                rec = sampler.decode(zsrc); edit = sampler.decode(xt)
            rec_pil, edit_pil = to_pil(rec, a.save_norm), to_pil(edit, a.save_norm)
            if rec_pil.size != pil.size:
                # 편집 해상도가 원본과 다르면 원본 크기로 되돌려 저장한다. evaluate.py 는 저장 캔버스의 우하단 512² 를 자르므로
                # 1024² 를 그대로 저장하면 편집본의 1/4 만 채점되는 오류가 난다 (2026-09-24 감사에서 발견).
                rec_pil = rec_pil.resize(pil.size, Image.LANCZOS); edit_pil = edit_pil.resize(pil.size, Image.LANCZOS)
            save_pair([rec_pil, edit_pil], out)
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
