"""모든 방법에 같은 마스크 처리와 저장 형식을 적용하기 위한 공용 함수.

prepare_latent_mask 는 DirectEdit inversion/flow_direct_correction_inv_sd35.py 의 prepare_mask 와 동일하다:
  테두리 0 패딩(이미지가 16의 배수가 아닐 때) -> 3px 팽창(7x7 사각 커널) -> ToTensor -> bilinear 로 latent 해상도 축소.
마스크 값 1 = 편집 영역(생성), 0 = 배경(원본 latent 로 대체).
"""
import os
import numpy as np, cv2, torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def prepare_latent_mask(mask_image, latent_hw, device, dtype, dilation_pixels=3):
    lat_h, lat_w = latent_hw
    mask = mask_image.convert("L")
    padded = Image.new("L", (lat_w * 8, lat_h * 8), 0)
    padded.paste(mask, (0, 0))
    if dilation_pixels > 0:
        k = 2 * int(dilation_pixels) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        padded = Image.fromarray(cv2.dilate(np.array(padded), kernel, iterations=1))
    t = transforms.ToTensor()(padded).to(device).to(dtype)
    return F.interpolate(t.unsqueeze(0), size=(lat_h, lat_w), mode="bilinear", align_corners=False)


def load_mask(mask_dir, stem, image_size):
    p = os.path.join(mask_dir, stem + "_mask.png")
    if not os.path.exists(p):
        raise FileNotFoundError("mask missing: " + p)
    m = Image.open(p).convert("L")
    if m.size != image_size:
        m = m.resize(image_size, Image.NEAREST)
    return m


def save_pair(images, stem, offset_ratio=0.02):
    """[재구성|편집] 두 장을 가로로 이어 <stem>_edited.png 로 저장한다 (DirectEdit view_images 와 같은 형식).
    images: PIL.Image 또는 HxWx3 uint8 ndarray 의 리스트. evaluate.py 는 우하단 512² 를 잘라 편집본으로 쓴다."""
    arrs = [np.asarray(im).astype(np.uint8) for im in images]
    h, w, _ = arrs[0].shape
    off = int(h * offset_ratio)
    canvas = np.ones((h, w * len(arrs) + off * (len(arrs) - 1), 3), np.uint8) * 255
    for j, a in enumerate(arrs):
        x0 = j * (w + off)
        canvas[:, x0:x0 + w] = a
    Image.fromarray(canvas).save(stem + "_edited.png")
