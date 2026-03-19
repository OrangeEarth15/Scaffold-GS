/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once

#include <iostream>
#include <vector>
#include <cstdint>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* means2D;
		float* cov3D;
		float4* conic_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P, bool inference_mode = false);
	};

	struct ImageState
	{
		uint2* ranges;
		uint16_t* n_contrib;  // Changed from uint32_t: max gaussians per pixel rarely exceeds 65535
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t pixel_count, size_t tile_count);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	template<typename T>
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}

	// Helper for GeometryState with inference_mode
	inline size_t required_geometry(size_t P, bool inference_mode)
	{
		char* size = nullptr;
		GeometryState::fromChunk(size, P, inference_mode);
		return ((size_t)size) + 128;
	}

	// Helper for ImageState which needs both pixel_count and tile_count
	inline size_t required_image(size_t pixel_count, size_t tile_count)
	{
		char* size = nullptr;
		ImageState::fromChunk(size, pixel_count, tile_count);
		return ((size_t)size) + 128;
	}
};