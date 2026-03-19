"""
Scaffold-GS inference VRAM profiling - BASELINE VERSION (原始未优化)

这是未应用任何优化的原始版本，用于对比测试。

把 generate_neural_gaussians + render 的逻辑拆开内联，
在每个关键步骤之间插入 torch.cuda.memory_allocated() 检查点，
观测真实推理过程中的显存变化。

原始版本特征：
- 未优化1: 先创建完整的 [G, 3], [G, 7], [G, 22] 等大张量，再做 mask 过滤
- 未优化2: MLP 使用 FP32 精度（未利用 Tensor Core FP16 加速）
- 未优化3: BinningState 使用 4 个独立 buffer（CUDA 层）

性能预期（1.68M 高斯点场景）：
- Generate 峰值：~1,190.7 MB
- Rasterize 峰值：~1,117.0 MB
- 渲染速度：~37.45 FPS

对应测试数据：profiling/results/garden_detailed_pure_inference.txt

Usage:
  python render_profile_baseline.py -m <model_path> --skip_train --num_frames 1
"""
import os
import torch
import numpy as np
import subprocess
import math
import json
import time
from dataclasses import dataclass, field
from typing import List

# ── 自动选择最空闲的 GPU ──
# 查询每块 GPU 已用显存，选最小的那张，设为 CUDA_VISIBLE_DEVICES
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

# ── 项目依赖 ──
# 需要在 CUDA_VISIBLE_DEVICES 设置之后再 import，否则 torch 可能已锁定 GPU
from scene import Scene
from gaussian_renderer import GaussianModel, prefilter_voxel
# 直接导入 rasterizer 底层组件，因为我们要把 render() 拆开内联
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from argparse import ArgumentParser
# einops.repeat 用于 anchor 属性的 repeat 展开 (V→G)
from einops import repeat


# ===================================================================
# 显存监测工具函数
# ===================================================================

def mb(b: int) -> str:
    """把字节数格式化为 MB 字符串，保留 1 位小数"""
    return f"{b / 1024**2:.1f}"


def gpu_mem():
    """
    读取 PyTorch CUDA 显存的三个核心指标:
      - allocated:  当前所有活跃 tensor 占用的显存（真实占用）
      - reserved:   caching allocator 从 CUDA 申请的总池大小
                    （≈ nvidia-smi 看到的值，>= allocated）
      - peak:       自上次 reset_peak_memory_stats() 以来 allocated 的历史最高值
    """
    a = torch.cuda.memory_allocated()
    r = torch.cuda.memory_reserved()
    p = torch.cuda.max_memory_allocated()
    return a, r, p


def log_mem(tag: str):
    """
    打印一行显存状态。格式:
      [tag.............................]  alloc=  XXX.X MB  reserved=  XXX.X MB  peak=  XXX.X MB
    tag 用 '.' 右填充到 50 字符，三个数值右对齐 9 字符。
    """
    a, r, p = gpu_mem()
    print(f"  [{tag:.<50s}]  alloc={mb(a):>9s} MB  reserved={mb(r):>9s} MB  peak={mb(p):>9s} MB")


# ===================================================================
# 核心: 单帧 profiling
# ===================================================================

def profile_one_frame(view, gaussians, pipeline, background):
    """
    对单帧推理做细粒度显存追踪。
    把原始的 prefilter_voxel() + render() 拆成子步骤，
    在每个子步骤前后调用 log_mem() 记录显存变化。

    Args:
        view:       单个相机视角 (Camera 对象)
        gaussians:  加载好的 GaussianModel
        pipeline:   PipelineParams (含 debug 等标志)
        background: 背景色 tensor [3], on CUDA
    """

    pc = gaussians   # 短别名, 与原始 generate_neural_gaussians 代码变量名一致
    pipe = pipeline

    # 确保之前所有 GPU 操作完成，显存状态稳定
    torch.cuda.synchronize()
    # 重置峰值计数器，后续 max_memory_allocated() 只反映本帧
    torch.cuda.reset_peak_memory_stats()

    print("=" * 90)
    # 此时 GPU 上只有模型参数 (常驻基底，不含 GT 图片)
    log_mem("0. baseline (model only, no images)")

    # =================================================================
    # Phase 1: prefilter_voxel — 视锥体剔除
    # 对应 gaussian_renderer/__init__.py 的 prefilter_voxel()
    # 对所有 N 个 anchor 做投影，返回 bool mask 标记哪些可见
    # 内部会临时分配 GeometryState(N) + ImageState(H*W)，返回后释放
    # =================================================================
    torch.cuda.reset_peak_memory_stats()   # Phase 1 单独追踪峰值
    voxel_visible_mask = prefilter_voxel(view, pc, pipe, background)
    torch.cuda.synchronize()
    log_mem("1. after prefilter_voxel")
    # 打印可见 anchor 统计，用于验证理论分析中 visible_ratio 的假设
    n_visible = voxel_visible_mask.sum().item()
    n_total = voxel_visible_mask.shape[0]
    print(f"       anchors: {n_total:,} total, {n_visible:,} visible ({n_visible/n_total*100:.1f}%)")

    # =================================================================
    # Phase 2: generate_neural_gaussians (内联拆分版)
    # 对应 gaussian_renderer/__init__.py 第 18-111 行
    # 原始代码是一个函数调用，这里拆开以便在每个子步骤间插入显存检查点
    # =================================================================

    # 清空 caching allocator 的空闲块，让 reserved 降到 allocated
    # 这样后续的显存增长从干净基线开始，更容易观察增量
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()   # Phase 2 单独追踪峰值
    log_mem("2a. before generate (after cache clear)")

    visible_mask = voxel_visible_mask
    is_training = False

    # ── Step 2b: 用 bool mask 提取可见 anchor 的属性 ──
    # 对应原函数第 23-26 行
    # [visible_mask] 是 bool indexing，会创建新 tensor (copy, 不是 view)
    # V = visible anchor 数量
    feat = pc._anchor_feat[visible_mask]      # [V, feat_dim=32]
    anchor = pc.get_anchor[visible_mask]       # [V, 3]
    grid_offsets = pc._offset[visible_mask]    # [V, n_offsets, 3]
    grid_scaling = pc.get_scaling[visible_mask] # [V, 6]
    torch.cuda.synchronize()
    log_mem("2b. extract visible anchors")

    # ── Step 2c: 计算视角方向 & 拼接 MLP 输入 ──
    # 对应原函数第 29-54 行
    ob_view = anchor - view.camera_center          # [V, 3] 从 anchor 指向相机
    ob_dist = ob_view.norm(dim=1, keepdim=True)    # [V, 1] 到相机的距离
    ob_view = ob_view / ob_dist                     # [V, 3] 归一化方向向量

    # 可选: feature bank (多分辨率特征加权)
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1)

    # 拼接 MLP 输入: [feat(32) + ob_view(3) + ob_dist(1)] = [V, 36]
    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)
    # 不含距离的版本: [feat(32) + ob_view(3)] = [V, 35]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)
    # 可选: appearance embedding (per-camera)
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * view.uid
        appearance = pc.get_appearance(camera_indicies)
    torch.cuda.synchronize()
    log_mem("2c. view features + cat")

    # ── Step 2d: MLP 推理 opacity + 生成 mask ──
    # 对应原函数第 57-68 行
    # mlp_opacity 输入 [V, 35 or 36], 输出 [V, n_offsets]
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view)
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)
    # reshape 到 [G, 1], G = V * n_offsets (每个 anchor 生成 n_offsets 个候选)
    neural_opacity = neural_opacity.reshape([-1, 1])
    # Tanh 输出范围 [-1, 1], 阈值 0: opacity > 0 的才保留
    mask = (neural_opacity > 0.0)
    mask = mask.view(-1)                    # [G] bool
    opacity = neural_opacity[mask]          # [S, 1], S = 存活数
    n_survived = mask.sum().item()          # S: 最终进入 rasterizer 的高斯点数
    n_generated = mask.shape[0]             # G: mask 前的总候选数
    torch.cuda.synchronize()
    log_mem("2d. MLP opacity + mask")
    # 打印存活比例，用于验证 opacity_survive_ratio 假设
    print(f"       generated: {n_generated:,}, survived: {n_survived:,} ({n_survived/n_generated*100:.1f}%)")

    # ── Step 2e: MLP 推理 color ──
    # 对应原函数第 71-81 行
    # mlp_color 输入 [V, 35/36 + appearance_dim], 输出 [V, 3*n_offsets]
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0] * pc.n_offsets, 3])  # [G, 3]
    torch.cuda.synchronize()
    log_mem("2e. MLP color")

    # ── Step 2f: MLP 推理 covariance (scale + rotation) ──
    # 对应原函数第 84-88 行
    # mlp_cov 输出 [V, 7*n_offsets], reshape 为 [G, 7] (3 scale + 4 quaternion)
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0] * pc.n_offsets, 7])
    torch.cuda.synchronize()
    log_mem("2f. MLP cov (scale+rot)")

    # ── Step 2g: 大合并 — generate 阶段的显存峰值 ──
    # 对应原函数第 91-96 行
    # 把 scaling(6) + anchor(3) + color(3) + scale_rot(7) + offsets(3) = 22 维
    # 合并成一个 [G, 22] 的大张量，再统一做 mask 过滤
    # 此时所有之前的中间张量都还活着 (Python 局部变量持有引用)
    offsets = grid_offsets.view([-1, 3])                                         # view, 无新分配
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)                     # [V, 9]
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)  # [G, 9]
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)  # [G, 22] ← 大张量
    torch.cuda.synchronize()
    log_mem("2g. concatenated_all [G,22] ★ GEN PEAK")

    # ── Step 2h: mask 过滤 + 后处理 ──
    # 对应原函数第 97-106 行
    masked = concatenated_all[mask]         # [S, 22], bool indexing → copy
    # 按列拆分回各属性
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    # 最终 scale: anchor 后 3 维 scaling * sigmoid(MLP 输出的前 3 维)
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3])
    # 最终 rotation: 四元数归一化
    rot = pc.rotation_activation(scale_rot[:,3:7])
    # 最终 position: anchor 位置 + offset * anchor 前 3 维 scaling
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets
    torch.cuda.synchronize()
    log_mem("2h. post-process (xyz/color/opacity/scaling/rot)")

    # 记录 Phase 2 的峰值 (从 2a 的 reset 到现在的最高 allocated)
    gen_peak = torch.cuda.max_memory_allocated()

    # ── 释放中间张量 ──
    # 原始 generate_neural_gaussians() 返回时，局部变量自动出作用域释放
    # 这里因为内联了代码，需要手动 del
    # 只保留 5 个输出: xyz, color, opacity, scaling, rot
    del feat, anchor, grid_offsets, grid_scaling, ob_view, ob_dist
    del cat_local_view, cat_local_view_wodist, neural_opacity
    del concatenated, concatenated_repeated, concatenated_all, masked
    del scaling_repeat, repeat_anchor
    if pc.appearance_dim > 0:
        del appearance, camera_indicies
    torch.cuda.synchronize()
    # 此时 alloc 应骤降 (中间张量已释放), 但 reserved 可能仍高 (caching pool 未归还)
    log_mem("2i. after del intermediates")

    # 强制把 caching pool 的空闲块归还给 CUDA
    # 这样 Phase 3 从干净基线开始
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    log_mem("2j. after empty_cache")

    # =================================================================
    # Phase 3: CUDA Rasterizer
    # 对应 gaussian_renderer/__init__.py render() 第 128-166 行
    # 进入 diff-gaussian-rasterization 的 CUDA kernel:
    #   1. 分配 GeometryState(P) — preprocess (投影、协方差、conic)
    #   2. prefix sum tiles_touched → 得到 R (num_rendered)
    #   3. 分配 BinningState(R) — 排序用 ← 显存最大头
    #   4. 分配 ImageState(H*W) — per-pixel 累加 buffer
    #   5. 逐 tile alpha blending → out_color
    # =================================================================
    torch.cuda.reset_peak_memory_stats()   # Phase 3 单独追踪峰值

    # screenspace_points: 2D 投影坐标, 训练时用于梯度, 推理时无实际作用
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    # 相机参数
    tanfovx = math.tan(view.FoVx * 0.5)
    tanfovy = math.tan(view.FoVy * 0.5)

    # 构造 rasterizer 配置
    raster_settings = GaussianRasterizationSettings(
        image_height=int(view.image_height),
        image_width=int(view.image_width),
        tanfovx=tanfovx, tanfovy=tanfovy,
        bg=background,
        scale_modifier=1.0,
        viewmatrix=view.world_view_transform,   # 世界→相机变换矩阵
        projmatrix=view.full_proj_transform,     # 完整投影矩阵
        sh_degree=1,                             # Scaffold-GS 不用 SH, 但接口要求传
        campos=view.camera_center,
        prefiltered=False,                       # 已做过 prefilter, 但这里仍为 False (CUDA 侧会再检查)
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    log_mem("3a. before rasterize")

    # 调用 CUDA rasterizer:
    #   means3D: 高斯中心 [S, 3]
    #   means2D: 屏幕空间坐标 (由 CUDA 内部计算)
    #   shs=None: 不用球谐, 直接传预计算的 RGB
    #   colors_precomp: MLP 输出的 RGB [S, 3]
    #   opacities: [S, 1]
    #   scales: [S, 3], rotations: [S, 4]
    # 内部会分配 GeometryState + BinningState + ImageState
    # BinningState 是显存峰值的绝对主因 (R × 32 bytes)
    rendered_image, radii = rasterizer(
        means3D=xyz, means2D=screenspace_points,
        shs=None, colors_precomp=color,
        opacities=opacity, scales=scaling,
        rotations=rot, cov3D_precomp=None)

    torch.cuda.synchronize()
    # rasterizer 返回后, CUDA 内部的 geomBuffer/binningBuffer/imgBuffer
    # 已出作用域释放, 但 peak 记录了中间的最高水位
    rast_peak = torch.cuda.max_memory_allocated()
    log_mem("3b. after rasterize ★ RAST PEAK")

    # =================================================================
    # 全部清理
    # =================================================================
    del xyz, color, opacity, scaling, rot, screenspace_points, radii
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # 应回到 baseline (只剩模型参数), 如果没回来说明有泄漏
    log_mem("4. after full cleanup")

    # ── 汇总打印 ──
    print()
    print(f"  ★ generate peak (alloc):   {mb(gen_peak):>9s} MB")    # Phase 2 最高水位
    print(f"  ★ rasterize peak (alloc):  {mb(rast_peak):>9s} MB")   # Phase 3 最高水位
    print(f"  ★ overall peak (alloc):    {mb(max(gen_peak, rast_peak)):>9s} MB")  # 全局峰值
    print(f"  Resolution: {view.image_width}x{view.image_height}")
    print(f"  P (rasterized Gaussians): {n_survived:,}")             # 最终渲染的高斯点数
    print("=" * 90)

    return rendered_image


# ===================================================================
# 入口
# ===================================================================

if __name__ == "__main__":
    # ── 命令行参数 ──
    parser = ArgumentParser(description="VRAM profiling script")
    # ModelParams: 模型超参 (feat_dim, n_offsets, appearance_dim 等)
    # sentinel=True: 允许从训练保存的 cfg_args 自动恢复, 不必全部手动指定
    model = ModelParams(parser, sentinel=True)
    # PipelineParams: 管线参数 (debug, compute_cov3D_python 等)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)          # -1 = 加载最新 checkpoint
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--num_frames", default=3, type=int,          # 只 profile 几帧 (不需要跑全部)
                        help="Number of frames to profile (default: 3)")
    # get_combined_args: 合并命令行参数 + model_path 下保存的 cfg_args
    args = get_combined_args(parser)

    # 设置随机种子 + quiet 模式
    safe_state(args.quiet)

    # torch.no_grad(): 关闭 autograd, 推理时不需要梯度
    # 省去 MLP forward 中间激活值的存储, 减少显存开销
    with torch.no_grad():
        # 构造 GaussianModel (anchor + offset + MLP 结构)
        gaussians = GaussianModel(
            args.feat_dim, args.n_offsets, args.voxel_size,
            args.update_depth, args.update_init_factor, args.update_hierachy_factor,
            args.use_feat_bank, args.appearance_dim, args.ratio,
            args.add_opacity_dist, args.add_cov_dist, args.add_color_dist)

        # Scene: 加载数据集 + 相机参数 + 模型权重 (ply + MLP checkpoints)
        dataset = model.extract(args)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        # eval 模式: 关闭 dropout/BatchNorm 训练行为
        gaussians.eval()

        # 背景色: 白底或黑底
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 优先用 test cameras, 没有则 fallback 到 train cameras
        views = scene.getTestCameras()
        if not views:
            views = scene.getTrainCameras()

        # 纯推理模式: 删除所有 GT 图片，只保留相机位姿
        # 这样基线显存才反映真实业务场景 (不含训练数据)
        all_cams = list(scene.getTrainCameras()) + list(scene.getTestCameras())
        for v in all_cams:
            if hasattr(v, 'original_image'):
                v.original_image = None
            if hasattr(v, 'gt_alpha_mask'):
                v.gt_alpha_mask = None
        del all_cams

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # 打印模型加载后的基线显存 (纯推理，不含 GT 图片)
        log_mem("MODEL LOADED (images freed)")

        num = min(args.num_frames, len(views))
        print(f"\nProfiling {num} frames...\n")

        # 逐帧 profile, 每帧独立输出完整的显存时间线
        for i in range(num):
            print(f"\n>>> Frame {i} (camera uid={views[i].uid})")
            profile_one_frame(views[i], gaussians, pipeline.extract(args), background)
