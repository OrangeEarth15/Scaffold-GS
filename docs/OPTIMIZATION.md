# Scaffold-GS 显存优化

## 一、优化总览

记录 Scaffold-GS 推理阶段的显存优化，包括 PyTorch Generate 阶段和 CUDA Rasterize 阶段。

**测试场景**:
- 数据集: MipNeRF360 Garden
- 分辨率: 2594×1680
- 高斯点数: 1.68M (visible)
- 锚点数: 933,874 (可见 338,724)

**优化结果**:

| 指标 | 说明 | Baseline | Optimized | 提升 |
|------|------|----------|-----------|------|
| **显存优化** | PyTorch + CUDA 全栈 | | | |
| └ Generate峰值 | Early Pruning + FP16 | 1,190.7 MB | 670.8 MB | -519.9 MB (-43.7%) |
| └ Rasterize峰值 | CUDA 优化 | 1,117.0 MB | 894.4 MB | -222.6 MB (-19.9%) |
| **FP16 加速** | MLP 混合精度 | | | |
| └ 渲染速度 | FP32 vs FP16 | 39.22 FPS | 60.97 FPS | +55.4% |
| └ 图像质量 | PSNR | FP32 Baseline | 76.66 dB | 无损 |

---

## 二、Generate 阶段优化 (PyTorch)

### 优化 G1: Early Opacity Pruning

**节省**: -362.9 MB

原始代码为所有候选点（3.39M）计算 color 和 scale_rot，但 opacity mask 会过滤掉约 50%。这些被过滤点的内存分配和计算都是浪费。

**文件**: `gaussian_renderer/__init__.py`

**Before** (先分配完整张量，最后过滤):
```python
# 为所有 3.39M 候选点计算
color = pc.get_color_mlp(cat_local_view).reshape([G, 3])      # [3.39M, 3]
scale_rot = pc.get_cov_mlp(cat_local_view).reshape([G, 7])    # [3.39M, 7]
offsets = grid_offsets.view([-1, 3])                          # [3.39M, 3]

# 拼接成巨大的 [G, 22] 张量
concatenated_all = torch.cat([
    concatenated_repeated, color, scale_rot, offsets
], dim=-1)  # [3.39M, 22] = 300+ MB

# 最后才过滤
masked = concatenated_all[mask]  # [1.7M, 22]
```

**After** (opacity mask 后立即过滤):
```python
# 生成 mask 后立即过滤，避免分配 G 级别的大张量
mask = (neural_opacity > 0.0).view(-1)
opacity = neural_opacity[mask]  # [1.7M, 1]

color = pc.get_color_mlp(cat_local_view)
color = color.reshape([G, 3])[mask]  # 直接过滤 → [1.7M, 3]

scale_rot = pc.get_cov_mlp(cat_local_view)
scale_rot = scale_rot.reshape([G, 7])[mask]  # 直接过滤 → [1.7M, 7]

offsets = grid_offsets.view([-1, 3])[mask]  # [1.7M, 3]

# 不再创建 concatenated_all，避免 [G, 22] 的大张量
```

**结果**: Generate 峰值 1,190.7 MB → 827.8 MB (-362.9 MB, -30.5%)

---

### 优化 G2: FP16 Mixed Precision

**速度提升**: +55.8%

MLP 前向推理使用 FP32 计算，未充分利用 Tensor Core 的 FP16 加速能力。

**文件**: `gaussian_renderer/__init__.py`

**Before**:
```python
# FP32 计算
neural_opacity = pc.get_opacity_mlp(cat_local_view)
color = pc.get_color_mlp(...)
scale_rot = pc.get_cov_mlp(...)
```

**After** (使用 PyTorch AMP):
```python
# FP16 Mixed Precision
with torch.amp.autocast('cuda', enabled=True):
    neural_opacity = pc.get_opacity_mlp(cat_local_view)  # FP16 计算
    color = pc.get_color_mlp(...)
    scale_rot = pc.get_cov_mlp(...)

    # mask 和 reshape 操作
    mask = (neural_opacity > 0.0).view(-1)
    opacity = neural_opacity[mask]
    color = color.reshape([G, 3])[mask]
    scale_rot = scale_rot.reshape([G, 7])[mask]

# 转回 FP32 供 rasterizer 使用
opacity = opacity.float()
color = color.float()
scale_rot = scale_rot.float()
```

**结果**:
- 渲染速度: 53.79 FPS → 83.80 FPS (+55.8%)
- 图像质量: PSNR 76.66 dB (远超 50 dB 无损阈值)
- 类型转换开销: ~0.18 ms/frame (可忽略)

---

### 优化 G3: Conditional Concatenation

**节省**: ~10-20 MB

**Before**:
```python
# 总是拼接 distance，即使某些 MLP 不需要
cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [V, 36]
```

**After**:
```python
# 检查哪些 MLP 需要 distance
need_dist = pc.add_opacity_dist or pc.add_color_dist or pc.add_cov_dist

# 基础特征（总是需要）
cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)  # [V, 35]

# 只在需要时才拼接 distance
if need_dist:
    cat_local_view = torch.cat([cat_local_view_wodist, ob_dist], dim=1)
else:
    cat_local_view = None  # 节省 ~10 MB
```

---

### 优化 G4: In-place Operations

**节省**: ~5-10 MB

使用 in-place 操作减少临时 tensor 分配：

```python
# Before
xyz = repeat_anchor + offsets * scaling_repeat[:,:3]

# After (in-place)
offsets.mul_(scaling_repeat[:,:3])  # in-place multiplication
xyz = repeat_anchor.add(offsets)    # avoid creating temp with +
```

---

## 三、Rasterize 阶段优化 (CUDA)

### 优化 R1: BinningState In-place Sorting

**节省**: ~100 MB (理论)

原始代码为 tile-Gaussian pair 排序分配了 4 个独立 buffer (24P bytes):
- `point_list_keys` + `point_list_keys_unsorted` (16P bytes)
- `point_list` + `point_list_unsorted` (8P bytes)

**文件**: `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu`

**Before**:
```cpp
// 分配 4 个独立 buffer
obtain(chunk, binning.point_list_keys, P, 128);
obtain(chunk, binning.point_list, P, 128);
obtain(chunk, binning.point_list_keys_unsorted, P, 128);
obtain(chunk, binning.point_list_unsorted, P, 128);

// 排序：unsorted → sorted
cub::DeviceRadixSort::SortPairs(
    binning.list_sorting_space, binning.sorting_size,
    binning.point_list_keys_unsorted, binning.point_list_keys,
    binning.point_list_unsorted, binning.point_list,
    num_rendered);
```

**After** (只分配 2 个 buffer，in-place sorting):
```cpp
// 只分配 2 个 buffer
obtain(chunk, binning.point_list_keys, P, 128);
obtain(chunk, binning.point_list, P, 128);

// 指针复用（unsorted = sorted）
binning.point_list_keys_unsorted = binning.point_list_keys;
binning.point_list_unsorted = binning.point_list;

// CUB in-place sorting (input buffer = output buffer)
cub::DeviceRadixSort::SortPairs(
    binning.list_sorting_space, binning.sorting_size,
    binning.point_list_keys, binning.point_list_keys,  // 同一 buffer
    binning.point_list, binning.point_list,            // 同一 buffer
    num_rendered);
```

**结果**: 内存分配 24P → 12P bytes (-50%)

---

### 优化 R2: ImageState Ranges Over-allocation 修复

**节省**: -34.7 MB

`ranges` buffer 错误地按 pixel 数量分配，实际只需要按 tile 数量分配。

- 错误: 分配 4,357,920 个 uint2 (像素数)
- 正确: 分配 17,115 个 uint2 (tile 数)
- 浪费: 254倍 over-allocation

**文件**: `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h`

**Before**:
```cpp
struct ImageState {
    uint2* ranges;
    uint32_t* n_contrib;
    float* accum_alpha;

    static ImageState fromChunk(char*& chunk, size_t N);  // N = pixel_count
};

// 调用时：
ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);
```

**After**:
```cpp
struct ImageState {
    uint2* ranges;          // Tile-based
    uint16_t* n_contrib;    // Pixel-based (2 bytes)
    float* accum_alpha;     // Pixel-based

    static ImageState fromChunk(char*& chunk, size_t pixel_count, size_t tile_count);
};

// 调用时：
size_t tile_count = tile_grid.x * tile_grid.y;  // 17,115
ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height, tile_count);
```

**文件**: `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu`

**Before**:
```cpp
ImageState ImageState::fromChunk(char*& chunk, size_t N) {
    ImageState img;
    obtain(chunk, img.accum_alpha, N, 128);
    obtain(chunk, img.n_contrib, N, 128);
    obtain(chunk, img.ranges, N, 128);  // BUG: N = pixel_count
    return img;
}
```

**After**:
```cpp
ImageState ImageState::fromChunk(char*& chunk, size_t pixel_count, size_t tile_count) {
    ImageState img;
    obtain(chunk, img.accum_alpha, pixel_count, 128);
    obtain(chunk, img.n_contrib, pixel_count, 128);
    obtain(chunk, img.ranges, tile_count, 128);  // FIXED: tile_count
    return img;
}
```

**结果**: ranges 内存 34.8 MB → 0.137 MB (-34.7 MB)

同时将 n_contrib 改为 uint16_t (2 bytes) 节省 8.7 MB，总计 -43.4 MB。

---

### 优化 R3: 推理模式不分配 backward-only buffers

**节省**: -45 MB

`clamped` 和 `cov3D` 只在 backward pass 需要，推理时完全不需要分配。

**文件**: `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h`

**Before**:
```cpp
struct GeometryState {
    bool* clamped;    // 训练时记录 SH 颜色 clamp 状态
    float* cov3D;     // 训练时 backward 需要
    // ...

    static GeometryState fromChunk(char*& chunk, size_t P);
};
```

**After**:
```cpp
struct GeometryState {
    bool* clamped;
    float* cov3D;
    // ...

    static GeometryState fromChunk(char*& chunk, size_t P, bool inference_mode = false);
};
```

**文件**: `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu`

**Before**:
```cpp
GeometryState GeometryState::fromChunk(char*& chunk, size_t P) {
    GeometryState geom;
    obtain(chunk, geom.depths, P, 128);
    obtain(chunk, geom.clamped, P * 3, 128);    // 总是分配
    obtain(chunk, geom.cov3D, P * 6, 128);      // 总是分配
    // ...
}
```

**After**:
```cpp
GeometryState GeometryState::fromChunk(char*& chunk, size_t P, bool inference_mode) {
    GeometryState geom;
    obtain(chunk, geom.depths, P, 128);

    if (!inference_mode) {
        obtain(chunk, geom.clamped, P * 3, 128);
        obtain(chunk, geom.cov3D, P * 6, 128);
    } else {
        geom.clamped = nullptr;  // 推理时不分配
        geom.cov3D = nullptr;
    }
    // ...
}
```

**文件**: `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`

**Before**:
```cpp
// clamped 无条件写入
clamped[3 * idx + 0] = (result.x < 0);
clamped[3 * idx + 1] = (result.y < 0);
clamped[3 * idx + 2] = (result.z < 0);

// cov3D 写入 global memory
computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
cov3D = cov3Ds + idx * 6;
```

**After**:
```cpp
// clamped 检查 nullptr
if (clamped != nullptr) {
    clamped[3 * idx + 0] = (result.x < 0);
    clamped[3 * idx + 1] = (result.y < 0);
    clamped[3 * idx + 2] = (result.z < 0);
}

// cov3D 使用 local buffer (inference) 或 global (training)
float local_cov3D[6];
if (cov3Ds != nullptr) {
    computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
    cov3D = cov3Ds + idx * 6;
} else {
    computeCov3D(scales[idx], scale_modifier, rotations[idx], local_cov3D);
    cov3D = local_cov3D;  // Stack-allocated, 无 global memory 开销
}
```

**Inference Mode 自动检测**:
```cpp
// forward() 函数中自动检测
bool inference_mode = (shs == nullptr && colors_precomp != nullptr);
size_t chunk_size = required_geometry(P, inference_mode);
GeometryState geomState = GeometryState::fromChunk(chunkptr, P, inference_mode);
```

**结果**:
- clamped: -5 MB (1.68M × 3 bytes)
- cov3D: -40 MB (1.68M × 24 bytes)
- 总计: -45 MB

---

## 四、优化效果

**测试场景**: MipNeRF360 Garden (2594×1680, ~1.68M visible gaussians)

### 4.1 显存优化 (PyTorch + CUDA)

| 阶段 | Baseline | Optimized | 节省 |
|------|----------|-----------|------|
| Generate 峰值 | 1,190.7 MB | 670.8 MB | -519.9 MB (-43.7%) |
| Rasterize 峰值 | 1,117.0 MB | 894.4 MB | -222.6 MB (-19.9%) |

**测试数据**: `profiling/results/memory/baseline.txt` vs `optimized.txt`

**Generate 优化** (PyTorch, G1-G4):
- Early Opacity Pruning: 提前过滤低 opacity 点,避免为 3.39M 候选点分配巨大张量
- FP16 Mixed Precision: MLP 推理使用 FP16,降低中间 tensor 显存
- Conditional Concatenation: 按需拼接 distance 特征
- In-place Operations: 减少临时 tensor 分配

**Rasterize 优化** (CUDA, R1-R3):
- In-place Sorting: CUB 排序复用 buffer (分配量 24P → 12P bytes)
- Ranges Over-allocation 修复: 修正 254x 过度分配 (-34.7 MB)
- uint16_t for n_contrib: 类型窄化 4 bytes → 2 bytes (-8.7 MB)
- Inference Mode: 不分配 backward-only buffers (-45 MB)

### 4.2 FP16 混合精度加速

**测试说明**: FP32 vs FP16 MLP 推理对比 (均基于优化后 CUDA)

| 指标 | FP32 | FP16 | 提升 |
|------|------|------|------|
| FPS | 39.22 | 60.97 | +55.4% |
| ms/frame | 25.50 | 16.40 | -35.7% |

**测试数据**: `profiling/results/speedup/benchmark.txt`

### 4.3 FP16 质量验证

**测试说明**: 5 个测试视角 FP32 vs FP16 渲染质量对比

| 指标 | 结果 |
|------|------|
| PSNR | 76.66 dB ± 0.74 dB |
| SSIM | 1.0000 |
| 质量影响 | 无损 (远超 50 dB 阈值) |

**测试数据**: `profiling/results/precision/validation.txt`

---

## 五、失败的尝试

### 5.1 Anchor FP16 存储

**方案**: 将 anchor 参数存为 FP16 节省显存

**结果**:
- 存储减半 (252.9 → 126.5 MB)
- 运行时显存反增 +793.5 MB
- PSNR 降至 39.46 dB (不可接受)

**原因**: 计算需 FP32，产生转换副本开销；FP16 精度不足导致 Gaussian 几何误差累积。

### 5.2 Tiled Rendering

**方案**: 图像分块渲染降低峰值

**结果**: 显存增加 13.5%

**原因**: 需为每块重复执行 prefilter 和 Generate；块间 Gaussian 重叠导致总处理量增加。

### 5.3 gsplat Packed

**方案**: 使用 gsplat packed 格式

**结果**: 不适用

**原因**: Packed 优化需 batch_size > 1，Scaffold-GS 推理是单帧 (batch_size = 1)。

---

## 六、编译说明

CUDA 优化需要重新编译：

```bash
cd submodules/diff-gaussian-rasterization
pip install -e . --no-build-isolation
```

---

## 七、文件清单

### 6.1 优化代码

**Generate 阶段 (PyTorch)**:
- `gaussian_renderer/__init__.py` (行 56-107)
  - Early Opacity Pruning
  - FP16 Mixed Precision
  - Conditional Concatenation
  - In-place Operations

**Rasterize 阶段 (CUDA)**:
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h`
  - ImageState 接口修改 (ranges, n_contrib 类型)
  - GeometryState inference_mode 参数
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu`
  - BinningState in-place sorting
  - ImageState 修复 ranges allocation
  - GeometryState 条件分配 (clamped, cov3D)
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`
  - clamped nullptr 检查
  - cov3D local buffer
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.h`
  - render() 签名 (uint16_t n_contrib)
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.h/backward.cu`
  - render() 签名更新

### 6.2 测试脚本

**显存测试**:
- `profiling/profile_memory_baseline.py` - Baseline 详细分析
- `profiling/profile_memory_optimized.py` - 优化后详细分析

**速度测试**:
- `profiling/profile_speed.py` - FP16 vs FP32 速度对比

**精度测试**:
- `profiling/profile_precision.py` - FP16 图像质量验证

### 6.3 测试结果

**显存测试** (`profiling/results/memory/`):
- `baseline.txt` - 优化前显存 profiling (Generate: 1190.7 MB, Rasterize: 1117.0 MB)
- `optimized.txt` - 优化后显存 profiling (Generate: 670.8 MB, Rasterize: 894.4 MB)

**速度测试** (`profiling/results/speedup/`):
- `benchmark.txt` - FP16 vs FP32 速度对比 (39.22 → 60.97 FPS)

**精度测试** (`profiling/results/precision/`):
- `validation.txt` - FP16 精度验证详细日志 (PSNR 76.66 dB, SSIM 1.0000)
