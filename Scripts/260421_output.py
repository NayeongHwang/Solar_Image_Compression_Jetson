#!/usr/bin/env python3
import os, sys, io, math, warnings
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from astropy.io import fits
from astropy.visualization import ImageNormalize, SqrtStretch
import astropy.units as u
from sunpy.map import Map
from pytorch_msssim import ssim
from compressai.zoo import cheng2020_attn
import matplotlib.pyplot as plt

# 환경 설정
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
warnings.filterwarnings("ignore")

# AIA 시각화 설정
AIA_CMAP = "sdoaia211"
AIA_NORM = ImageNormalize(stretch=SqrtStretch(), vmin=0, vmax=5000)
OUT_DIR = Path("~/_data/results").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

def calc_psnr(x, y):
    mse = np.mean((x - y) ** 2)
    return 20 * np.log10(1.0 / np.sqrt(mse + 1e-12))

def tensor_to_np(img):
    return img.squeeze().cpu().numpy()

# ─────────────────────────────────────────────
# FITS 로드 (Log Scaling 적용)
# ─────────────────────────────────────────────
def load_fits(path: Path, target_size: int = 1024):
    smap = Map(str(path))
    if smap.data.shape[0] != target_size:
        smap = smap.resample((target_size, target_size) * u.pix)

    raw_dn = smap.data.astype(np.float32)
    
    # [개선] 로그 변환으로 다이내믹 레인지 압축 (격자 방지 핵심)
    # 0 이하의 값은 0으로 클리핑 후 log(1+x) 적용
    data_log = np.log1p(np.maximum(raw_dn, 0))
    
    d_min, d_max = data_log.min(), data_log.max()
    data_norm = (data_log - d_min) / (d_max - d_min + 1e-12)
    
    tensor = torch.from_numpy(data_norm).unsqueeze(0).unsqueeze(0)
    return tensor, raw_dn, d_min, d_max

# ─────────────────────────────────────────────
# CompressAI (Padding 및 Log 역변환 적용)
# ─────────────────────────────────────────────
def run_compressai(model, x, d_min, d_max):
    x_dev = x.cuda()
    if x_dev.shape[1] == 1:
        x_dev = x_dev.repeat(1, 3, 1, 1)

    # 64배수 패딩 (가장 안전한 constant 모드로 변경)
    h, w = x_dev.shape[2:]
    p = 64
    pad_h = (p - h % p) % p
    pad_w = (p - w % p) % p
    
    # mode='edge' 대신 'constant' 사용
    x_padded = F.pad(x_dev, (0, pad_w, 0, pad_h), mode='constant', value=0)

    with torch.no_grad():
        out = model(x_padded)

    # 패딩 제거 및 1채널 추출 (원본 크기 h, w로 슬라이싱)
    x_hat = out["x_hat"][:, 0:1, :h, :w].cpu()
    
    # BPP 계산
    num_pix = h * w
    bpp = (torch.log(out["likelihoods"]["y"]).sum() + 
           torch.log(out["likelihoods"]["z"]).sum()) / (-math.log(2) * num_pix)

    # 메트릭 계산
    psnr = calc_psnr(x[:, 0:1, :, :].numpy(), x_hat.numpy())
    ssim_val = ssim(x_hat, x, data_range=1.0).item()

    # [역변환] 로그 스페이스 -> DN 스페이스
    recon_log = tensor_to_np(x_hat) * (d_max - d_min) + d_min
    recon_dn = np.expm1(recon_log)

    return {
        "label": "CompressAI (cheng2020_attn q=6)",
        "recon_dn": recon_dn,
        "psnr": psnr,
        "ssim": ssim_val,
        "bpp": bpp.item(),
    }
# ─────────────────────────────────────────────
# 시각화 및 저장
# ─────────────────────────────────────────────
def save_result(raw_dn, recon_dn, info):
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0d0d0d")
    
    # 태양 물리 데이터 시각화 표준 적용
    im = ax.imshow(recon_dn, origin="lower", cmap=AIA_CMAP, norm=AIA_NORM)
    
    title = f"{info['label']}\nPSNR={info['psnr']:.2f}dB  SSIM={info['ssim']:.4f}  BPP={info['bpp']:.4f}"
    ax.set_title(title, color="white", fontsize=12, pad=10)
    ax.axis("off")
    
    plt.tight_layout()
    fpath = OUT_DIR / "recon_optimized.png"
    plt.savefig(fpath, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓ 결과 저장 완료: {fpath}")

def main():
    fits_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("~/_data/aia211.fits").expanduser()
    if not fits_path.exists():
        print(f"파일을 찾을 수 없습니다: {fits_path}")
        return

    print(f"데이터 로딩 및 로그 변환: {fits_path.name}")
    x, raw_dn, d_min, d_max = load_fits(fits_path)

    print("모델 로딩 중 (cheng2020_attn)...")
    model = cheng2020_attn(quality=6, pretrained=True).eval().cuda()

    print("압축 및 복원 실행 중...")
    result = run_compressai(model, x, d_min, d_max)

    print(f"\n결과 요약:")
    print(f"PSNR: {result['psnr']:.2f} dB (Log space)")
    print(f"SSIM: {result['ssim']:.4f}")
    print(f"BPP : {result['bpp']:.4f}")

    save_result(raw_dn, result['recon_dn'], result)

if __name__ == "__main__":
    main()
