#!/usr/bin/env python3
"""
FP16 精度验证 - 对比 FP32 vs FP16 渲染质量

使用 monkey patch 控制 FP16，避免修改源文件
"""
import os
import sys
import torch
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Select least busy GPU
import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

# Monkey patch to control FP16
_original_autocast = torch.amp.autocast

def controlled_autocast(device_type, enabled=True, *args, **kwargs):
    global _use_fp16
    return _original_autocast(device_type, enabled=_use_fp16, *args, **kwargs)

torch.amp.autocast = controlled_autocast

from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.image_utils import psnr
from torchvision.utils import save_image


def calculate_ssim(img1, img2):
    """Calculate SSIM between two images [C, H, W] in range [0, 1]"""
    from math import exp

    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(img1, img2, window, window_size, channel):
        mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    (_, channel, _, _) = img1.size()
    window_size = 11
    window = create_window(window_size, channel).to(img1.device)
    return ssim(img1, img2, window, window_size, channel)


def render_with_precision(model_path, use_fp16, num_views=5):
    """Render images with specified precision (FP16 or FP32)"""
    global _use_fp16
    _use_fp16 = use_fp16

    sys.argv = ['profile_precision.py', '-m', model_path]

    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--iteration', default=-1, type=int)
    args = get_combined_args(parser)

    print(f"\nLoading model from {model_path}...")
    with torch.no_grad():
        gaussians = GaussianModel(
            args.feat_dim, args.n_offsets, args.voxel_size,
            args.update_depth, args.update_init_factor, args.update_hierachy_factor,
            args.use_feat_bank, args.appearance_dim, args.ratio,
            args.add_opacity_dist, args.add_cov_dist, args.add_color_dist)

        dataset = model.extract(args)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device='cuda')

        views = scene.getTestCameras()[:num_views]
        pipe_args = pipeline.extract(args)

        rendered_images = []
        print(f"\nRendering {len(views)} views with FP16={'enabled' if use_fp16 else 'disabled'}...")
        for view in tqdm(views, desc="Rendering"):
            result = render(view, gaussians, pipe_args, background)
            rendered_image = result["render"]
            rendered_images.append(rendered_image.cpu())

        return rendered_images


def compare_renders(fp32_images, fp16_images):
    """Compare FP32 vs FP16 rendered images using PSNR and SSIM"""
    psnr_values = []
    ssim_values = []

    print("\nComparing FP32 vs FP16 images...")
    for i, (img_fp32, img_fp16) in enumerate(zip(fp32_images, fp16_images)):
        img_fp32 = img_fp32.unsqueeze(0).cuda()
        img_fp16 = img_fp16.unsqueeze(0).cuda()

        psnr_val = psnr(img_fp32, img_fp16).item()
        ssim_val = calculate_ssim(img_fp32, img_fp16).item()

        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)

        print(f"  View {i}: PSNR = {psnr_val:.2f} dB, SSIM = {ssim_val:.4f}")

    return psnr_values, ssim_values


def main():
    parser = ArgumentParser(description="FP16 precision validation")
    parser.add_argument("--model_path", "-m", required=True, type=str)
    parser.add_argument("--num_views", "-n", type=int, default=5)
    args = parser.parse_args()

    print("=" * 80)
    print("FP16 Precision Validation")
    print("=" * 80)

    # Render with FP32
    print("\n[1/3] Rendering with FP32 (baseline)...")
    fp32_images = render_with_precision(args.model_path, use_fp16=False, num_views=args.num_views)

    # Render with FP16
    print("\n[2/3] Rendering with FP16 (optimized)...")
    fp16_images = render_with_precision(args.model_path, use_fp16=True, num_views=args.num_views)

    # Compare
    print("\n[3/3] Comparing quality metrics...")
    psnr_values, ssim_values = compare_renders(fp32_images, fp16_images)

    # Summary
    print("\n" + "=" * 80)
    print("Results Summary")
    print("=" * 80)
    print(f"\nPSNR (FP32 vs FP16):")
    print(f"  Average: {np.mean(psnr_values):.2f} dB ± {np.std(psnr_values):.2f} dB")
    print(f"  Min:     {np.min(psnr_values):.2f} dB")
    print(f"  Max:     {np.max(psnr_values):.2f} dB")

    print(f"\nSSIM (FP32 vs FP16):")
    print(f"  Average: {np.mean(ssim_values):.4f} ± {np.std(ssim_values):.4f}")
    print(f"  Min:     {np.min(ssim_values):.4f}")
    print(f"  Max:     {np.max(ssim_values):.4f}")

    print("\n" + "=" * 80)
    print("Interpretation")
    print("=" * 80)
    if np.mean(psnr_values) > 50:
        print("✓ FP16 precision is LOSSLESS (PSNR > 50 dB)")
    elif np.mean(psnr_values) > 40:
        print("✓ FP16 precision is HIGH QUALITY (PSNR > 40 dB)")
    else:
        print("⚠ FP16 may have visible quality loss (PSNR < 40 dB)")

    if np.mean(ssim_values) > 0.99:
        print("✓ FP16 structural similarity is EXCELLENT (SSIM > 0.99)")
    elif np.mean(ssim_values) > 0.95:
        print("✓ FP16 structural similarity is GOOD (SSIM > 0.95)")
    else:
        print("⚠ FP16 may have structural differences (SSIM < 0.95)")

    print("=" * 80 + "\n")

    # Save results
    results = {
        'psnr': {
            'values': psnr_values,
            'mean': float(np.mean(psnr_values)),
            'std': float(np.std(psnr_values)),
            'min': float(np.min(psnr_values)),
            'max': float(np.max(psnr_values))
        },
        'ssim': {
            'values': ssim_values,
            'mean': float(np.mean(ssim_values)),
            'std': float(np.std(ssim_values)),
            'min': float(np.min(ssim_values)),
            'max': float(np.max(ssim_values))
        }
    }

    output_path = Path('profiling/renders/fp16_validation_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
