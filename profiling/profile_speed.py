#!/usr/bin/env python3
"""
Comprehensive FP16 vs FP32 speed comparison
Temporarily modifies autocast to test both modes
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import numpy as np
import subprocess
from argparse import ArgumentParser

# Select least busy GPU
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

def measure_speed(model_path, use_fp16=True, num_iterations=30, warmup=10):
    """Measure rendering speed"""
    global _use_fp16
    _use_fp16 = use_fp16

    sys.argv = ['benchmark_fp16_vs_fp32.py', '-m', model_path]

    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--iteration', default=-1, type=int)
    args = get_combined_args(parser)

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

        view = scene.getTestCameras()[0]
        pipe_args = pipeline.extract(args)

        # Warmup
        for i in range(warmup):
            result = render(view, gaussians, pipe_args, background)
        torch.cuda.synchronize()

        # Measure
        start_time = time.time()
        for i in range(num_iterations):
            result = render(view, gaussians, pipe_args, background)
        torch.cuda.synchronize()
        end_time = time.time()

        elapsed = end_time - start_time
        avg_time = elapsed / num_iterations
        fps = 1.0 / avg_time

        return avg_time, fps, result['render'].shape, result['visibility_filter'].sum().item()

if __name__ == "__main__":
    parser = ArgumentParser(description="FP16 vs FP32 benchmark")
    parser.add_argument("--model_path", "-m", required=True, type=str)
    parser.add_argument("--iterations", "-n", type=int, default=30)
    parser.add_argument("--warmup", "-w", type=int, default=10)
    args = parser.parse_args()

    print("=" * 80)
    print("FP16 vs FP32 Comprehensive Speed Comparison")
    print("=" * 80)
    print(f"\nModel: {args.model_path}")
    print(f"Iterations: {args.iterations} (Warmup: {args.warmup})")

    # Test FP32
    print("\n[1/2] Testing FP32 Baseline...")
    avg_time_fp32, fps_fp32, shape, num_gaussians = measure_speed(
        args.model_path, use_fp16=False,
        num_iterations=args.iterations, warmup=args.warmup
    )

    # Test FP16
    print("[2/2] Testing FP16 Mixed Precision...")
    avg_time_fp16, fps_fp16, _, _ = measure_speed(
        args.model_path, use_fp16=True,
        num_iterations=args.iterations, warmup=args.warmup
    )

    # Calculate speedup
    speedup = fps_fp16 / fps_fp32
    time_reduction = (1 - avg_time_fp16 / avg_time_fp32) * 100

    print(f"\n{'='*80}")
    print("Results Summary")
    print(f"{'='*80}")
    print(f"\nScene Info:")
    print(f"  Resolution: {shape[2]}x{shape[1]}")
    print(f"  Visible Gaussians: {num_gaussians:,}")

    print(f"\nFP32 Baseline:")
    print(f"  Time per frame: {avg_time_fp32*1000:.2f} ms")
    print(f"  FPS: {fps_fp32:.2f}")

    print(f"\nFP16 Optimized:")
    print(f"  Time per frame: {avg_time_fp16*1000:.2f} ms")
    print(f"  FPS: {fps_fp16:.2f}")

    print(f"\nSpeedup:")
    print(f"  FPS improvement: {speedup:.2f}x ({(speedup-1)*100:.1f}% faster)")
    print(f"  Time reduction: {time_reduction:.1f}%")

    print(f"\n{'='*80}\n")
