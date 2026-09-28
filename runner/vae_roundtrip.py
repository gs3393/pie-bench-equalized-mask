"""VAE 인코딩-디코딩 왕복본을 만든다.

마스크 밖이 원본 latent로 대체되면 배경 지표의 상한은 이 왕복 손실이 된다.
논문은 34.38 dB(700장)로 보고한다. 같은 부분집합에서 직접 재서 비교한다.
출력 형식은 run_condition.py 와 같아 evaluate.py 로 그대로 채점된다.
"""
import argparse, os, torch
import PIL.Image as Image

from mmdit.sd35_pipeline import StableDiffusion3Pipeline
from inversion.flow_direct_correction_inv_sd35 import Accurate_Inversion_SD3
from inversion.inv_utils import load_PIE_images, view_images

ap = argparse.ArgumentParser()
ap.add_argument("--src_path", required=True)
ap.add_argument("--saved_path", required=True)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16).to(args.device)
pipe.vae.eval()
inv = Accurate_Inversion_SD3(pipe, 30, args.device, 1.0, 2.0, 0, args.saved_path)

_, _, imgs, _, _, _ = load_PIE_images(args.src_path, edit_category_list=[str(i) for i in range(10)])
os.makedirs(args.saved_path, exist_ok=True)
for i, f in enumerate(imgs):
    stem = os.path.splitext(os.path.basename(f))[0]
    out = os.path.join(args.saved_path, stem)
    if os.path.exists(out + "_edited.png"):
        continue
    size = Image.open(f).convert("RGB").size
    z = inv.encode_latent(f)
    rt = inv.latent2image(z, size)
    view_images([rt, rt], out)
    if (i + 1) % 50 == 0:
        print("%d/%d" % (i + 1, len(imgs)), flush=True)
print("roundtrip done:", len(imgs))
