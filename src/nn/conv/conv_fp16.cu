// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include <vector>
#include <cuda.h>
#include <assert.h>

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <unordered_map>
#include <nvrtc.h>
#include <algorithm>

#include "cudakernel/nn/conv/conv_fp16.h"
#include "cudakernel/nn/conv/gene_kernel.h"
#include "cudakernel/nn/conv/group_padding.h"
#include "cudakernel/common/cuda_check.h"
#include "kernel_type.h"
#include "conv_common.h"
#include "conv_jit.h"
#include "cuda_nvrtc.h"
#include "common/init_lut.h"
#include "common/merge_split.h"

#include "float.h"

#include "cutlass/library/operation_table.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/library/library.h"
#include "cutlass/library/singleton.h"

#define TIMES 4

#define SPK_KPARAM_LIST                                    \
    pad_input,                                             \
        d_flt,                                             \
        conv_out,                                          \
        kloop_num,                                         \
        in_lut, in_lut_size,                               \
        flt_lut, flt_lut_size,                             \
        num_chl_per_spk_head, num_chl_per_spk_tail,        \
        in_hw, out_hw,                                     \
        flt_hw, splitk,                                    \
        conv_param.in_height, conv_param.in_width,         \
        conv_param.in_num, conv_param.num_grp,             \
        num_chl_per_grp, num_chl_per_grp_pad,              \
        conv_param.flt_height, conv_param.flt_width,       \
        num_flt_per_grp, num_flt_per_grp_pad,              \
        conv_param.out_height, conv_param.out_width,       \
        conv_param.stride_height, conv_param.stride_width, \
        conv_param.pad_height, conv_param.pad_width,       \
        conv_param.hole_height, conv_param.hole_width,     \
        conv_param.has_bias, (int *)bias

#define LUT_KPARAM_LIST                                               \
    pad_input,                                                        \
        d_flt,                                                        \
        conv_out,                                                     \
        kloop_num,                                                    \
        in_lut, in_lut_size,                                          \
        flt_lut, flt_lut_size,                                        \
        in_hw, out_hw,                                                \
        flt_hw, splitk,                                               \
        conv_param.in_height, conv_param.in_width,                    \
        conv_param.in_num, conv_param.num_grp,                        \
        num_chl_per_grp, num_chl_per_grp_pad,                         \
        conv_param.flt_height, conv_param.flt_width,                  \
        num_flt_per_grp, num_flt_per_grp_pad,                         \
        conv_param.out_height, conv_param.out_width,                  \
        conv_param.stride_height, conv_param.stride_width,            \
        conv_param.pad_height, conv_param.pad_width,                  \
        conv_param.hole_height, conv_param.hole_width,                \
        conv_param.has_bias, bias,                                    \
        fuse_param.has_activation, clip_min,                          \
        fuse_param.has_clip, clip_max,                                \
        fuse_param.has_prelu, (const void *)fuse_param.prelu,         \
        fuse_param.has_elt, (const int4 *)fuse_param.pre_data,        \
        fuse_param.has_elt_activation, elt_clip_min,                  \
        fuse_param.has_elt_clip, elt_clip_max,                        \
        fuse_param.has_elt_prelu, (const void *)fuse_param.elt_prelu, \
        leaky, elt_leaky,                                             \
        fuse_param.has_concat, concat_offset_v8,                      \
        concat_stride_v8

#define SWZL_SPK_KPARAM_LIST                               \
        d_flt,                                             \
        pad_input,                                         \
        conv_out,                                          \
        kloop_num,                                         \
        in_lut, in_lut_size,                               \
        flt_lut, flt_lut_size,                             \
        num_chl_per_spk_head, num_chl_per_spk_tail,        \
        in_hw, out_hw,                                     \
        flt_hw, splitk,                                    \
        conv_param.in_height, conv_param.in_width,         \
        conv_param.in_num, conv_param.num_grp,             \
        num_chl_per_grp, num_chl_per_grp_pad,              \
        conv_param.flt_height, conv_param.flt_width,       \
        num_flt_per_grp, num_flt_per_grp_pad,              \
        conv_param.out_height, conv_param.out_width,       \
        conv_param.stride_height, conv_param.stride_width, \
        conv_param.pad_height, conv_param.pad_width,       \
        conv_param.hole_height, conv_param.hole_width,     \
        conv_param.has_bias, (int *)bias

#define SWZL_LUT_KPARAM_LIST                                          \
        d_flt,                                                        \
        pad_input,                                                    \
        conv_out,                                                     \
        kloop_num,                                                    \
        in_lut, in_lut_size,                                          \
        flt_lut, flt_lut_size,                                        \
        in_hw, out_hw,                                                \
        flt_hw, splitk,                                               \
        conv_param.in_height, conv_param.in_width,                    \
        conv_param.in_num, conv_param.num_grp,                        \
        num_chl_per_grp, num_chl_per_grp_pad,                         \
        conv_param.flt_height, conv_param.flt_width,                  \
        num_flt_per_grp, num_flt_per_grp_pad,                         \
        conv_param.out_height, conv_param.out_width,                  \
        conv_param.stride_height, conv_param.stride_width,            \
        conv_param.pad_height, conv_param.pad_width,                  \
        conv_param.hole_height, conv_param.hole_width,                \
        conv_param.has_bias, bias,                                    \
        fuse_param.has_activation, clip_min,                          \
        fuse_param.has_clip, clip_max,                                \
        fuse_param.has_prelu, (const void *)fuse_param.prelu,         \
        fuse_param.has_elt, (const int4 *)fuse_param.pre_data,        \
        fuse_param.has_elt_activation, elt_clip_min,                  \
        fuse_param.has_elt_clip, elt_clip_max,                        \
        fuse_param.has_elt_prelu, (const void *)fuse_param.elt_prelu, \
        leaky, elt_leaky,                                             \
        fuse_param.has_concat, concat_offset_v8,                      \
        concat_stride_v8

#define IDX_KPARAM_LIST                                               \
    pad_input,                                                        \
        d_flt,                                                        \
        conv_out,                                                     \
        kloop_num, koff_num_pad,                                      \
        in_hw, out_hw,                                                \
        flt_hw, out_nhw,                                              \
        conv_param.in_height, conv_param.in_width,                    \
        conv_param.in_num, conv_param.num_grp,                        \
        conv_param.num_chl, num_chl_per_grp,                          \
        in_chl_per_grp_pad, flt_chl_per_grp_pad,                      \
        conv_param.flt_height, conv_param.flt_width,                  \
        num_flt_per_grp, num_flt_per_grp_pad,                         \
        conv_param.out_height, conv_param.out_width,                  \
        conv_param.stride_height, conv_param.stride_width,            \
        conv_param.pad_height, conv_param.pad_width,                  \
        conv_param.hole_height, conv_param.hole_width,                \
        conv_param.has_bias, bias,                                    \
        fuse_param.has_activation, clip_min,                          \
        fuse_param.has_clip, clip_max,                                \
        fuse_param.has_prelu, (const void *)fuse_param.prelu,         \
        fuse_param.has_elt, (const int4 *)fuse_param.pre_data,        \
        fuse_param.has_elt_activation, elt_clip_min,                  \
        fuse_param.has_elt_clip, elt_clip_max,                        \
        fuse_param.has_elt_prelu, (const void *)fuse_param.elt_prelu, \
        leaky, elt_leaky,                                             \
        fuse_param.has_concat, concat_offset_v8,                      \
        concat_stride_v8

#define MERGE_KPARAM_LIST                                             \
    conv_out, final_out,                                              \
        spk_height_v1, spk_width_v8,                                  \
        out_hw, splitk *splitf,                                       \
        conv_param.has_bias, bias,                                    \
        fuse_param.has_activation, clip_min,                          \
        fuse_param.has_clip, clip_max,                                \
        fuse_param.has_prelu, (const void *)fuse_param.prelu,         \
        fuse_param.has_elt, (const int4 *)fuse_param.pre_data,        \
        fuse_param.has_elt_activation, elt_clip_min,                  \
        fuse_param.has_elt_clip, elt_clip_max,                        \
        fuse_param.has_elt_prelu, (const void *)fuse_param.elt_prelu, \
        leaky, elt_leaky,                                             \
        fuse_param.has_concat, concat_offset_v8,                      \
        concat_stride_v8

static std::vector<kernel_info_t> g_fp16_kvec;
static bool is_g_fp16_kvec_initialized = false;

static std::unordered_map<size_t, algo_param_t> g_conv_shape_hash;

__inline__ void InitializeFP16ConvKernelContainer(std::vector<kernel_info_t> &g_fp16_kvec, const cudaDeviceProp& device_prop, ppl::common::datatype_t type)
{
#ifndef PPLNN_ENABLE_CUDA_JIT
    if (type == ppl::common::DATATYPE_FLOAT16) {
        if (device_prop.major == 7 && device_prop.minor == 5) {
#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 10020
            // sm75 kernels
            Initialize2spkSM75FP16Hmma1688ConvF1KernelContainer(g_fp16_kvec);
            Initialize2spkSM75FP16Hmma1688ConvF3KernelContainer(g_fp16_kvec);
            Initialize2spkSM75FP16Hmma1688ConvFNKernelContainer(g_fp16_kvec);
            Initialize2spkSM75FP16Hmma1688ConvFSKernelContainer(g_fp16_kvec);

            InitializeIdxnSM75FP16Hmma1688ConvKernelContainer(g_fp16_kvec);

            InitializeSwzlSM75FP16Hmma1688ConvF1KernelContainer(g_fp16_kvec);
            InitializeSwzlSM75FP16Hmma1688ConvF3KernelContainer(g_fp16_kvec);
            InitializeSwzlSM75FP16Hmma1688ConvFNKernelContainer(g_fp16_kvec);
#endif
        } else if (device_prop.major > 8 || (device_prop.major == 8 && device_prop.minor >= 0)) {
#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 10020
            // sm75 kernels
            Initialize2spkSM75FP16Hmma1688ConvF1KernelContainer(g_fp16_kvec);
            Initialize2spkSM75FP16Hmma1688ConvF3KernelContainer(g_fp16_kvec);
            Initialize2spkSM75FP16Hmma1688ConvFNKernelContainer(g_fp16_kvec);
            Initialize2spkSM75FP16Hmma1688ConvFSKernelContainer(g_fp16_kvec);

            InitializeIdxnSM75FP16Hmma1688ConvKernelContainer(g_fp16_kvec);

            InitializeSwzlSM75FP16Hmma1688ConvF1KernelContainer(g_fp16_kvec);
            InitializeSwzlSM75FP16Hmma1688ConvF3KernelContainer(g_fp16_kvec);
            InitializeSwzlSM75FP16Hmma1688ConvFNKernelContainer(g_fp16_kvec);
#endif

#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 11000
            // sm80 kernels
            Initialize2spkSM80FP16Hmma1688ConvF1KernelContainer(g_fp16_kvec);
            Initialize2spkSM80FP16Hmma1688ConvF3KernelContainer(g_fp16_kvec);
            Initialize2spkSM80FP16Hmma1688ConvFNKernelContainer(g_fp16_kvec);
            Initialize2spkSM80FP16Hmma1688ConvFSKernelContainer(g_fp16_kvec);

            InitializeSwzlSM80FP16Hmma1688ConvF1KernelContainer(g_fp16_kvec);
            InitializeSwzlSM80FP16Hmma1688ConvF3KernelContainer(g_fp16_kvec);
            InitializeSwzlSM80FP16Hmma1688ConvFNKernelContainer(g_fp16_kvec);

            Initialize2spkSM80FP16Hmma16816ConvF1KernelContainer(g_fp16_kvec);
            Initialize2spkSM80FP16Hmma16816ConvF3KernelContainer(g_fp16_kvec);
            Initialize2spkSM80FP16Hmma16816ConvFNKernelContainer(g_fp16_kvec);
            Initialize2spkSM80FP16Hmma16816ConvFSKernelContainer(g_fp16_kvec);

            InitializeIdxnSM80FP16Hmma16816ConvKernelContainer(g_fp16_kvec);

            InitializeSwzlSM80FP16Hmma16816ConvF1KernelContainer(g_fp16_kvec);
            InitializeSwzlSM80FP16Hmma16816ConvF3KernelContainer(g_fp16_kvec);
            InitializeSwzlSM80FP16Hmma16816ConvFNKernelContainer(g_fp16_kvec);
#endif
        }
    }
#endif

    is_g_fp16_kvec_initialized = true;
}

std::string GetConvShapeString(const conv_param_t &conv_param)
{
    return std::string("b" + std::to_string(conv_param.in_num) +
                       "_c" + std::to_string(conv_param.num_chl) +
                       "_d" + std::to_string(conv_param.num_flt) +
                       "_g" + std::to_string(conv_param.num_grp) +
                       "_h" + std::to_string(conv_param.in_height) +
                       "_w" + std::to_string(conv_param.in_width) +
                       "_r" + std::to_string(conv_param.flt_height) +
                       "_s" + std::to_string(conv_param.flt_width) +
                       "_p" + std::to_string(conv_param.pad_height) +
                       "_q" + std::to_string(conv_param.pad_width) +
                       "_u" + std::to_string(conv_param.stride_height) +
                       "_v" + std::to_string(conv_param.stride_width) +
                       "_y" + std::to_string(conv_param.hole_height) +
                       "_x" + std::to_string(conv_param.hole_width) +
                       "_");
}

__inline__ size_t GetConvShapeHashKey(conv_param_t &conv_param)
{
    return std::hash<std::string>{}(GetConvShapeString(conv_param));
}

uint64_t PPLCUDAConvolutionGetCompilationBufSize(ppl::common::datatype_t type, conv_param_t &conv_param, uint64_t workspace)
{
    int pad_size = GetPadSize(type);

    uint32_t num_chl_per_grp = conv_param.num_chl / conv_param.num_grp;
    uint32_t num_flt_per_grp = conv_param.num_flt / conv_param.num_grp;

    uint32_t num_chl_per_grp_pad = Align(num_chl_per_grp, pad_size);
    uint32_t num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size);

    bool is_in_grp_pad  = num_chl_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;
    bool is_out_grp_pad = num_flt_per_grp_pad != num_flt_per_grp && conv_param.num_grp != 1;

    uint64_t cvt_input_size = 0;
    uint64_t cvt_output_size = 0;

    if (is_in_grp_pad)
        cvt_input_size = GetCvtInputSize(type, conv_param, num_chl_per_grp_pad);

    if (is_out_grp_pad)
        cvt_output_size = getCvtOutputSize(type, conv_param, num_flt_per_grp_pad);

    uint64_t split_size = GetMaxSplitSize(type, conv_param, num_flt_per_grp_pad);

    uint64_t total_size = cvt_input_size + cvt_output_size + split_size;

    return total_size <= workspace ? total_size : workspace;
}
uint64_t PPLCUDAConvolutionGetRuntimeBufSize(
    ppl::common::datatype_t type,
    conv_param_t &conv_param,
    unsigned int splitk,
    unsigned int splitf,
    uint64_t workspace)
{
    int pad_size = GetPadSize(type);

    uint32_t num_chl_per_grp = conv_param.num_chl / conv_param.num_grp;
    uint32_t num_flt_per_grp = conv_param.num_flt / conv_param.num_grp;

    uint32_t num_chl_per_grp_pad = Align(num_chl_per_grp, pad_size);
    uint32_t num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size);

    bool is_in_grp_pad  = num_chl_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;
    bool is_out_grp_pad = num_flt_per_grp_pad != num_flt_per_grp && conv_param.num_grp != 1;

    uint64_t cvt_input_size = 0;
    uint64_t cvt_output_size = 0;

    if (is_in_grp_pad)
        cvt_input_size = GetCvtInputSize(type, conv_param, num_chl_per_grp_pad);
    if (is_out_grp_pad)
        cvt_output_size = getCvtOutputSize(type, conv_param, num_flt_per_grp_pad);

    uint64_t split_size = 0;
    
    if(splitk > 1 || splitf > 1)
        split_size = GetSplitKFSize(type, conv_param, num_flt_per_grp_pad, splitk, splitf);

    uint64_t total_size = cvt_input_size + cvt_output_size + split_size;

    return total_size <= workspace ? total_size : workspace;
}


/* -----------------  cutlass kernel select api ------------------ */
#define CUDA_CHECK(condition)                                                               \
  for (cudaError_t _of_cuda_check_status = (condition); _of_cuda_check_status != cudaSuccess;) { \
    std::cout << "[ERROR] Check failed: " #condition " : " << cudaGetErrorString(_of_cuda_check_status) \
                << " (" << _of_cuda_check_status << ") "; \
    exit(1); \
  } \

inline size_t HashCombine(size_t lhs, size_t rhs) {
    return lhs ^ (rhs + 0x9e3779b9 + (lhs << 6U) + (lhs >> 2U));
}

bool IsWeakerAlginOperation(const cutlass::library::Operation* lhs,
                            const cutlass::library::Operation* rhs) {
    const char* lhs_name = lhs->description().name;
    const char* rhs_name = rhs->description().name;
    const size_t len = std::strlen(lhs_name);
    const size_t suffix_len = std::strlen("align8");
    if (std::strlen(rhs_name) != len) { return false; }
    if (len < suffix_len) { return false; }
    const size_t prefix_len = len - suffix_len;
    if (std::strncmp(lhs_name, rhs_name, prefix_len) != 0) { return false; }
    const auto& HasLegalSuffix = [&](const char* str) {
        if (std::strncmp(str + prefix_len, "align", std::strlen("align")) != 0) { return false; }
        const char align = str[len - 1];
        return align == '8' || align == '4' || align == '2' || align == '1';
    };
    if ((!HasLegalSuffix(lhs_name)) || (!HasLegalSuffix(rhs_name))) { return false; }
    return lhs_name[len - 1] < rhs_name[len - 1];
}

struct Conv2dOperationCacheKey {
    cutlass::library::ConvFunctionalKey functional_key;
    cutlass::library::Conv2dConfiguration configuraion;
    size_t alignment;
    Conv2dOperationCacheKey(cutlass::library::ConvFunctionalKey functional_key,
                            cutlass::library::Conv2dConfiguration configuraion,
                            cutlass::library::ConvArguments arguments)
        : functional_key(functional_key), configuraion(configuraion) {
        const auto IsStrideAligned = [&](const std::vector<int64_t>& stride, size_t n) {
        return std::all_of(stride.cbegin(), stride.cend(),
                            [&](const int64_t& s) { return s % n == 0; });
        };
        // CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.A) % kCudaAlignSize, 0);
        // CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.B) % kCudaAlignSize, 0);
        // CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.C) % kCudaAlignSize, 0);
        // CHECK_EQ(reinterpret_cast<uintptr_t>(arguments.D) % kCudaAlignSize, 0);
        const auto IsAligned = [&](size_t n) {
        return IsStrideAligned(configuraion.stride_a, n) && IsStrideAligned(configuraion.stride_b, n)
                && IsStrideAligned(configuraion.stride_c, n);
        };
        if (IsAligned(8)) alignment = 8;
        else if (IsAligned(4)) alignment = 4;
        else if (IsAligned(2)) alignment = 2;
        else alignment = 1;
    }
};

struct Conv2dProblemSizeHasher {
    size_t operator()(const cutlass::conv::Conv2dProblemSize& problem_size) const {
        size_t hash = 0;
        hash = HashCombine(hash, std::hash<int>()(problem_size.N));
        hash = HashCombine(hash, std::hash<int>()(problem_size.H));
        hash = HashCombine(hash, std::hash<int>()(problem_size.W));
        hash = HashCombine(hash, std::hash<int>()(problem_size.C));
        hash = HashCombine(hash, std::hash<int>()(problem_size.P));
        hash = HashCombine(hash, std::hash<int>()(problem_size.Q));
        hash = HashCombine(hash, std::hash<int>()(problem_size.K));
        hash = HashCombine(hash, std::hash<int>()(problem_size.R));
        hash = HashCombine(hash, std::hash<int>()(problem_size.S));
        hash = HashCombine(hash, std::hash<int>()(problem_size.pad_h));
        hash = HashCombine(hash, std::hash<int>()(problem_size.pad_w));
        hash = HashCombine(hash, std::hash<int>()(problem_size.stride_h));
        hash = HashCombine(hash, std::hash<int>()(problem_size.stride_w));
        hash = HashCombine(hash, std::hash<int>()(problem_size.dilation_h));
        hash = HashCombine(hash, std::hash<int>()(problem_size.dilation_w));
        hash = HashCombine(hash, std::hash<int>()(static_cast<int>(problem_size.mode)));
        hash = HashCombine(hash, std::hash<int>()(problem_size.split_k_slices));
        hash = HashCombine(hash, std::hash<int>()(problem_size.groups));
        return hash;
    }
};

struct Conv2dConfigurationHasher {
    size_t operator()(const cutlass::library::Conv2dConfiguration& configuraion) const {
        size_t hash = std::hash<int>()(static_cast<int>(configuraion.split_k_mode));
        hash = HashCombine(hash, Conv2dProblemSizeHasher()(configuraion.problem_size));
        for (const int64_t v : configuraion.stride_a) {
            hash = HashCombine(hash, std::hash<int64_t>()(v));
        }
        for (const int64_t v : configuraion.stride_b) {
            hash = HashCombine(hash, std::hash<int64_t>()(v));
        }
        for (const int64_t v : configuraion.stride_c) {
            hash = HashCombine(hash, std::hash<int64_t>()(v));
        }
        return hash;
    }
};

struct Conv2dOperationCacheKeyHasher {
    size_t operator()(const Conv2dOperationCacheKey& key) const {
        size_t hash = cutlass::library::ConvFunctionalKeyHasher()(key.functional_key);
        hash = HashCombine(hash, Conv2dConfigurationHasher()(key.configuraion));
        hash = HashCombine(hash, std::hash<size_t>()(key.alignment));
        return hash;
    }
};

inline bool operator==(const cutlass::library::Conv2dConfiguration& lhs,
                       const cutlass::library::Conv2dConfiguration& rhs) {
    return lhs.split_k_mode == rhs.split_k_mode && lhs.problem_size == rhs.problem_size
            && lhs.stride_a == rhs.stride_a && lhs.stride_b == rhs.stride_b
            && lhs.stride_c == rhs.stride_c;
}

inline bool operator==(const Conv2dOperationCacheKey& lhs, const Conv2dOperationCacheKey& rhs) {
    return lhs.functional_key == rhs.functional_key && lhs.configuraion == rhs.configuraion
            && lhs.alignment == rhs.alignment;
}


using CacheMap = std::unordered_map<Conv2dOperationCacheKey, const cutlass::library::Operation*,
                                    Conv2dOperationCacheKeyHasher>;
static std::unordered_map<int, CacheMap> cache;


const cutlass::library::Operation* FindConv2dOperation(
    cudaStream_t &stream, 
    cutlass::library::ConvFunctionalKey functional_key,
    const cutlass::library::Conv2dConfiguration& configuraion,
    const cutlass::library::ConvArguments& arguments, 
    int device_arch,
    void* workspace, 
    size_t workspace_size) 
{
    Conv2dOperationCacheKey cache_key(functional_key, configuraion, arguments);
    const cutlass::library::Operation* fastest_operation = nullptr;
    int dev = 0;
    const auto& device_cache = cache[dev];
    constexpr int turing_warmup_iters = 3;
    // constexpr int turing_iters = TIMES;   // if too short, the cpu overhead maybe be too large
    constexpr int turing_iters = 20;   // cutlass conv's cpu overhead is not stable, so the iters num can't be too small
    const auto& it = device_cache.find(cache_key);
    LOG(DEBUG) << "Start selecting";
    if (it != device_cache.end()) {
        LOG(DEBUG) << "find cache";
        fastest_operation = it->second;
        const size_t device_workspace_size = fastest_operation->get_device_workspace_size(&configuraion);
        const size_t host_workspace_size = fastest_operation->get_host_workspace_size(&configuraion);

        auto status = fastest_operation->can_implement(&configuraion, &arguments);
        if(status != cutlass::Status::kSuccess || device_workspace_size > workspace_size) {
            fastest_operation = nullptr;
            return fastest_operation;
        }
        std::vector<uint8_t> host_workspace(host_workspace_size, 0);
        if (fastest_operation->initialize(
            &configuraion, 
            host_workspace.data(), 
            workspace, 
            stream) != cutlass::Status::kSuccess) {
            fastest_operation = nullptr;
            return fastest_operation;
        }

    } else {
        cudaEvent_t start;
        cudaEvent_t end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        float minTime = FLT_MAX;
        const auto& operations_map_it =
            cutlass::library::Singleton::get().operation_table.conv2d_operations.find(functional_key);
        if (operations_map_it == cutlass::library::Singleton::get().operation_table.conv2d_operations.cend()) {
            fastest_operation = nullptr;
            return fastest_operation;
        }
        const cutlass::library::ConvOperationVectorMap& operations_map = operations_map_it->second;
        for (const auto& pair : operations_map) {
            std::map<std::string, const cutlass::library::Operation*, std::greater<std::string>> operations;
            for (auto operation : pair.second) operations.emplace(operation->description().name, operation);
            const cutlass::library::Operation* prev_operation = nullptr;
            for (const auto& name_operation : operations) {
                const cutlass::library::Operation* operation = name_operation.second;
                if (prev_operation != nullptr && IsWeakerAlginOperation(operation, prev_operation))
                    continue;
                if (operation->description().tile_description.minimum_compute_capability > device_arch
                    || operation->description().tile_description.maximum_compute_capability < device_arch) {
                    continue;
                }
                auto status = operation->can_implement(&configuraion, &arguments);  // lots of operation can't pass here;
                if (status != cutlass::Status::kSuccess) continue;
                const size_t host_workspace_size = operation->get_host_workspace_size(&configuraion);
                const size_t device_workspace_size = operation->get_device_workspace_size(&configuraion);
                if (device_workspace_size > workspace_size) continue;
                std::vector<uint8_t> host_workspace(host_workspace_size, 0);
                if (operation->initialize(
                    &configuraion, 
                    host_workspace.data(), 
                    workspace,
                    stream) != cutlass::Status::kSuccess) {
                    continue;
                }

                const auto Run = [&]() {
                    auto init_status = operation->initialize(&configuraion, host_workspace.data(),
                                                            workspace, stream);
                    if (init_status != cutlass::Status::kSuccess)
                        LOG(ERROR) << "init_status != cutlass::Status::kSuccess";
                    auto run_status = operation->run(&arguments, host_workspace.data(),
                                                    workspace, stream);
                    if (run_status != cutlass::Status::kSuccess)
                        LOG(ERROR) << "run_status != cutlass::Status::kSuccess";
                };
                CUDA_CHECK(cudaDeviceSynchronize());
                for (int i = 0; i < turing_warmup_iters; ++i) Run();
                CUDA_CHECK(cudaEventRecord(start, stream));
                for (int i = 0; i < turing_iters; ++i) Run();
                CUDA_CHECK(cudaEventRecord(end, stream));
                CUDA_CHECK(cudaEventSynchronize(end));
                float elapsed = 0;
                CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, end));
                prev_operation = operation;
                LOG(DEBUG) << "kernel is : " << operation->description().name << " -> " << elapsed/turing_iters;
                if ((fastest_operation == nullptr || elapsed * TIMES / turing_iters < minTime) && operation != nullptr) {
                    fastest_operation = operation;
                    minTime = elapsed * TIMES / turing_iters; 
                    cache[dev][cache_key] = fastest_operation;
                }
            }
        }
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }
    return fastest_operation;
}

/* -----------------  FP16 KERNEL ------------------ */

double PPLCUDAConvolutionSelectKernel(
    const cudaDeviceProp& device_prop,
    cudaStream_t &stream,
    ppl::common::datatype_t type,
    int4 *d_input,
    int4 *d_flt,
    int4 *d_output,
    int4 *bias,
    int4 *d_temp_buf,
    algo_param_t &algo_param,
    conv_param_t &conv_param,
    fuse_param_t &fuse_param,
    uint64_t workspace)
{
#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 9020
    float minTime = FLT_MAX;

    bool is_cutlass_kernel_available = fuse_param.has_activation == 0 && !fuse_param.has_clip && fuse_param.has_prelu == 0 &&\
        !fuse_param.has_elt && fuse_param.has_elt_activation == 0 && !fuse_param.has_elt_clip && fuse_param.has_elt_prelu == 0 &&\
        fuse_param.elt_leaky==0 && !fuse_param.has_concat && conv_param.num_grp == 1;
    if (is_cutlass_kernel_available) {
        int device_arch = device_prop.major * 10 + device_prop.minor;
        // nhwc
        const int n = conv_param.in_num;
        const int h = conv_param.in_height;
        const int w = conv_param.in_width;
        const int c = conv_param.num_chl_pad;
        // krsc
        const int k = conv_param.num_flt_pad;
        const int r = conv_param.flt_height;
        const int s = conv_param.flt_width;
        // npqk
        const int p = conv_param.out_height;
        const int q = conv_param.out_width;

        int pad_h = conv_param.pad_height;
        int pad_w = conv_param.pad_width;
        int stride_h = conv_param.stride_height;
        int stride_w = conv_param.stride_width;
        int dilation_h = conv_param.hole_height;
        int dilation_w = conv_param.hole_height;
        LOG(DEBUG) << "n: " << n << "\th: " << h << "\tw: " << w << "\tc: " << c;
        LOG(DEBUG) << "k: " << k << "\tr: " << r << "\ts: " << s;
        LOG(DEBUG) << "p: " << p << "\tq: " << q;
        LOG(DEBUG) << "pad_h: " << pad_h << "\tpad_w: " << pad_w;
        LOG(DEBUG) << "stride_h: " << stride_h << "\tstride_w: " << stride_w;
        LOG(DEBUG) << "dilation_h: " << dilation_h << "\tdilation_w: " << dilation_w;

        cutlass::library::ConvFunctionalKey functional_key(
            cutlass::library::Provider::kCUTLASS, cutlass::library::ConvKind::kFprop,
            cutlass::library::NumericTypeID::kF16, cutlass::library::LayoutTypeID::kTensorNHWC,
            cutlass::library::NumericTypeID::kF16, cutlass::library::LayoutTypeID::kTensorNHWC,
            cutlass::library::NumericTypeID::kF16, cutlass::library::LayoutTypeID::kTensorNHWC,
            cutlass::library::NumericTypeID::kF16, cutlass::library::NumericTypeID::kF16
        );

        void* x_ptr = d_input;
        void* w_ptr = d_flt;
        void* y_ptr = d_output;
        void* bias_ptr = nullptr;
        if (conv_param.has_bias == 1) bias_ptr = bias;
        void* workspace_ptr = d_temp_buf;

        cutlass::conv::Conv2dProblemSize problem_size(
            n, h, w, c, k, r, s, p, q, 
            pad_h, pad_w, stride_h, stride_w, 
            dilation_h, dilation_w,
            cutlass::conv::Mode::kCrossCorrelation 
        );
        cutlass::library::Conv2dConfiguration configuraion;
        configuraion.split_k_mode = cutlass::conv::SplitKMode::kSerial;
        configuraion.problem_size = problem_size;
        configuraion.stride_a = {c, w * c, h * w * c};
        configuraion.stride_b = {c, s * c, r * s * c};
        configuraion.stride_c = {0, 0, 0};
        cutlass::library::ConvArguments arguments;
        arguments.A = x_ptr;
        arguments.B = w_ptr;
        arguments.reordered_B = nullptr;
        arguments.C = bias_ptr;
        arguments.D = y_ptr;

        union SP {
            float f{};
            half h;
        };

        SP alpha;
        SP beta;
        alpha.h = static_cast<half>(1.0F);
        if (conv_param.has_bias != 1) beta.h = static_cast<half>(0.0F);
        else beta.h = static_cast<half>(1.0F);
    
        arguments.alpha = &alpha;
        arguments.beta = &beta;
        arguments.pointer_mode = cutlass::library::ScalarPointerMode::kHost;

        Conv2dOperationCacheKey cache_key(functional_key, configuraion, arguments);
        const cutlass::library::Operation* fastest_operation;
        int dev = 0;
        const auto& device_cache = cache[dev];
        constexpr int turing_warmup_iters = 3;
        // constexpr int turing_iters = TIMES;   // if too short, the cpu overhead would be too large
        constexpr int turing_iters = 20;   // cutlass conv's cpu overhead is not stable, so the iters num can't be too small
        cudaEvent_t start;
        cudaEvent_t end;
        const auto& it = device_cache.find(cache_key);

        fastest_operation = FindConv2dOperation(
            stream, 
            functional_key, 
            configuraion, 
            arguments, 
            device_arch,
            workspace_ptr,
            workspace
        );

        cudaEventCreate(&start);
        cudaEventCreate(&end);
        if (fastest_operation != nullptr) {
            const size_t device_workspace_size = fastest_operation->get_device_workspace_size(&configuraion);
            const size_t host_workspace_size = fastest_operation->get_host_workspace_size(&configuraion);

            auto status = fastest_operation->can_implement(&configuraion, &arguments);
            if (status != cutlass::Status::kSuccess) LOG(ERROR) << "status != cutlass::Status::kSuccess";
            if (device_workspace_size > workspace) LOG(ERROR) << "device_workspace_size > workspace";
            std::vector<uint8_t> host_workspace(host_workspace_size, 0);
            if (fastest_operation->initialize(
                &configuraion, 
                host_workspace.data(), 
                workspace_ptr,
                stream) != cutlass::Status::kSuccess) ;
            const auto Run = [&]() {
                auto init_status = fastest_operation->initialize(&configuraion, host_workspace.data(),
                                                        workspace_ptr, stream);
                if (init_status != cutlass::Status::kSuccess) LOG(ERROR) << "init_status != cutlass::Status::kSuccess";
                auto run_status = fastest_operation->run(&arguments, host_workspace.data(),
                                                workspace_ptr, stream);
                if (run_status != cutlass::Status::kSuccess) LOG(ERROR) << "run_status != cutlass::Status::kSuccess";
            };
            CUDA_CHECK(cudaDeviceSynchronize());
            for (int i = 0; i < turing_warmup_iters; ++i) Run();
            CUDA_CHECK(cudaEventRecord(start, stream));
            for (int i = 0; i < turing_iters; ++i) Run();
            CUDA_CHECK(cudaEventRecord(end, stream));
            CUDA_CHECK(cudaEventSynchronize(end));
            float elapsed = 0;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, end));
            minTime = elapsed * TIMES / turing_iters;
            algo_param.algo_type = "CutlassHConv";
            algo_param.algo_name = fastest_operation->description().name;
            algo_param.kid = -1;
            algo_param.splitk = device_workspace_size;  // use splitk to store workspace
            algo_param.splitf = 1;
        }
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }
    if(!is_cutlass_kernel_available) fuse_param = fuse_param_t{}; // reset to default
    LOG(DEBUG) << "algo_type:" << algo_param.algo_type << 
                " algo_name:" << algo_param.algo_name << 
                " splitk:" << algo_param.splitk <<
                " splitf:" << algo_param.splitf;
    LOG(DEBUG) << "minTime: " << minTime;

    if (!is_g_fp16_kvec_initialized) {
        InitializeFP16ConvKernelContainer(g_fp16_kvec, device_prop, type);
        if (g_fp16_kvec.empty()) {
          LOG(ERROR) << "Fp16 kernel should be compiled on cuda >= 10.2 and run on architeture >= sm_75";
          return ppl::common::RC_UNSUPPORTED;
        }
    }

    size_t conv_shape_hash = GetConvShapeHashKey(conv_param);

    std::unordered_map<size_t, algo_param_t>::const_iterator conv_shape_hash_iterator = g_conv_shape_hash.find(conv_shape_hash);

    // if (conv_shape_hash_iterator != g_conv_shape_hash.end()) {
    //     algo_param.algo_type = "TuringHMMAImpgemm";
    //     algo_param.kid    = conv_shape_hash_iterator->second.kid;
    //     algo_param.splitk = conv_shape_hash_iterator->second.splitk;
    //     algo_param.splitf = conv_shape_hash_iterator->second.splitf;
    //     algo_param.algo_name   = conv_shape_hash_iterator->second.algo_name;

    //     return ppl::common::RC_SUCCESS;
    // }

    int pad_size = GetPadSize(type);

    int num_chl_per_grp = conv_param.num_chl / conv_param.num_grp;
    int num_flt_per_grp = conv_param.num_flt / conv_param.num_grp;

    int num_chl_per_grp_pad = Align(num_chl_per_grp, pad_size);
    int num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size);

    int in_hw  = conv_param.in_height * conv_param.in_width;
    int flt_hw = conv_param.flt_height * conv_param.flt_width;
    int out_hw = conv_param.out_height * conv_param.out_width;

    int concat_offset_v8 = fuse_param.concat_offset / pad_size;
    int concat_stride_v8 = fuse_param.concat_stride / pad_size;

    bool is_in_grp_pad  = num_chl_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;
    bool is_out_grp_pad = num_flt_per_grp_pad != num_flt_per_grp && conv_param.num_grp != 1;

    uint64_t buf_off_v4 = 0;

    int4 *pad_input  = d_input;
    int4 *pad_output = d_output;

    if (is_in_grp_pad) {
        pad_input = d_temp_buf;
        buf_off_v4 += GetCvtInputSize(type, conv_param, num_chl_per_grp_pad) / (_4INT_TO_INT4_ * _INT_TO_4BYTE_);

        PPLCUDAConvolutionCvtInput(stream, pad_input, d_input, type, conv_param);
    }

    if (is_out_grp_pad) {
        pad_output = d_temp_buf + buf_off_v4;
        buf_off_v4 += getCvtOutputSize(type, conv_param, num_flt_per_grp_pad) / (_4INT_TO_INT4_ * _INT_TO_4BYTE_);
    }

    int4 *final_out = fuse_param.has_concat ? (int4 *)fuse_param.post_concat : pad_output;

    int4 *splitk_buf = d_temp_buf + buf_off_v4;

    __half2 clip_min     = __float2half2_rn(fuse_param.clip_min);
    __half2 clip_max     = __float2half2_rn(fuse_param.clip_max);
    __half2 elt_clip_min = __float2half2_rn(fuse_param.elt_clip_min);
    __half2 elt_clip_max = __float2half2_rn(fuse_param.elt_clip_max);
    __half leaky         = __float2half(fuse_param.leaky);
    __half elt_leaky     = __float2half(fuse_param.elt_leaky);

    // float minTime = FLT_MAX;

    float elapsed;
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    const int SPLITK_OPTIONS[] = {1, 2, 4, 8};

    for (unsigned int spk = 0; spk < 4; spk++) {
        unsigned int splitk = SPLITK_OPTIONS[spk];

        for (unsigned int kid = 0; kid < g_fp16_kvec.size(); kid++) {
            unsigned int splitf = (g_fp16_kvec[kid].ktype == CONV_2SPK_FS) ? flt_hw : 1;

            if (!g_fp16_kvec[kid].CheckKernelTypeFeasible(conv_param.flt_height, conv_param.flt_width, num_chl_per_grp, splitk))
                continue;

            if (!g_fp16_kvec[kid].CheckSMemSizeFeasible(device_prop))
                continue;

            if (!g_fp16_kvec[kid].CheckGpuArchFeasible(device_prop))
                continue;

            if (!g_fp16_kvec[kid].CheckSplitkFeasible(num_chl_per_grp, splitk))
                continue;

            if (!g_fp16_kvec[kid].CheckSplitfFeasible(splitf, splitk))
                continue;

            int4 *conv_out = (splitk > 1 || splitf > 1) ? splitk_buf : final_out;

            dim3 block_size, grid_size;

            block_size.x = g_fp16_kvec[kid].cta_size_in_thd;
            block_size.y = 1;
            block_size.z = 1;

            int smem_size = g_fp16_kvec[kid].smem_size;

            if(g_fp16_kvec[kid].ktype == CONV_SWZL_F1 || g_fp16_kvec[kid].ktype == CONV_SWZL_F3 || \
                    g_fp16_kvec[kid].ktype == CONV_SWZL_FN) {
                grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, g_fp16_kvec[kid].tile_n_per_cta);
                grid_size.y = DivUp(num_flt_per_grp_pad, g_fp16_kvec[kid].tile_m_per_cta);
            } else {
                grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, g_fp16_kvec[kid].tile_m_per_cta);
                grid_size.y = DivUp(num_flt_per_grp_pad, g_fp16_kvec[kid].tile_n_per_cta);
            }

            grid_size.z = conv_param.num_grp * splitk * splitf;

            // warm up
            for (int i = 0; i < 3; i++) {
                if (g_fp16_kvec[kid].ktype == CONV_IDXN_C2 || g_fp16_kvec[kid].ktype == CONV_IDXN_C4 ||
                    g_fp16_kvec[kid].ktype == CONV_IDXN_C32) {
                    int tile_k_per_step = g_fp16_kvec[kid].tile_k_per_step;

                    int img_pad_size = pad_size;
                    int flt_pad_size = g_fp16_kvec[kid].flt_pad_size;
                    int out_nhw      = out_hw * conv_param.in_num;

                    int in_chl_per_grp_pad  = Align(num_chl_per_grp, img_pad_size);
                    int flt_chl_per_grp_pad = Align(num_chl_per_grp, flt_pad_size);
                    int num_flt_per_grp_pad = Align(num_flt_per_grp, img_pad_size);

                    int kloop_num    = DivUp(flt_hw * flt_chl_per_grp_pad, g_fp16_kvec[kid].tile_k_per_cta);
                    int koff_num_pad = Align(kloop_num * (g_fp16_kvec[kid].tile_k_per_cta / flt_pad_size), WARP_SIZE);

                    (g_fp16_kvec[kid].idx_kptr)<<<grid_size, block_size, 0, stream>>>(IDX_KPARAM_LIST);
                } else if (g_fp16_kvec[kid].ktype == CONV_2SPK_F1 || g_fp16_kvec[kid].ktype == CONV_2SPK_F3 ||
                           g_fp16_kvec[kid].ktype == CONV_2SPK_FN || g_fp16_kvec[kid].ktype == CONV_2SPK_FS ||
                           g_fp16_kvec[kid].ktype == CONV_SWZL_F1 || g_fp16_kvec[kid].ktype == CONV_SWZL_F3 ||
                           g_fp16_kvec[kid].ktype == CONV_SWZL_FN) {

                    int kloop_num = (flt_hw / splitf) * DivUp(num_chl_per_grp_pad, g_fp16_kvec[kid].tile_k_per_cta);

                    lut_t in_lut, flt_lut;
                    int in_lut_size, flt_lut_size;

                    InitializeInputLut(in_lut_size, in_lut.idx, conv_param.flt_height, conv_param.flt_width, conv_param.in_height, conv_param.in_width, conv_param.pad_height, conv_param.pad_width, conv_param.hole_height, conv_param.hole_width, num_chl_per_grp_pad, conv_param.num_grp, g_fp16_kvec[kid].tile_k_per_cta, pad_size);

                    InitializeFilterLut(flt_lut_size, flt_lut.idx, conv_param.flt_height, conv_param.flt_width, num_chl_per_grp_pad, g_fp16_kvec[kid].tile_k_per_cta, pad_size);

                    if (splitk == 1) {
                        g_fp16_kvec[kid].AdaptLutKernelSMemSize();

                        if(g_fp16_kvec[kid].ktype == CONV_SWZL_F1 || g_fp16_kvec[kid].ktype == CONV_SWZL_F3 || g_fp16_kvec[kid].ktype == CONV_SWZL_FN)
                            (g_fp16_kvec[kid].lut_kptr)<<<grid_size, block_size, smem_size, stream>>>(SWZL_LUT_KPARAM_LIST);
                        else {
                            (g_fp16_kvec[kid].lut_kptr)<<<grid_size, block_size, smem_size, stream>>>(LUT_KPARAM_LIST);
                        }
                    } else {
                        int num_chl_per_spk_head, num_chl_per_spk_tail;

                        InitializeNumChlPerSpk(num_chl_per_spk_head, num_chl_per_spk_tail, conv_param.num_chl, conv_param.num_grp, pad_size, g_fp16_kvec[kid].tile_k_per_cta, splitk);

                        g_fp16_kvec[kid].AdaptSpkKernelSMemSize();

                        if(g_fp16_kvec[kid].ktype == CONV_SWZL_F1 || g_fp16_kvec[kid].ktype == CONV_SWZL_F3 || g_fp16_kvec[kid].ktype == CONV_SWZL_FN)
                            (g_fp16_kvec[kid].spk_kptr)<<<grid_size, block_size, smem_size, stream>>>(SWZL_SPK_KPARAM_LIST);
                        else
                            (g_fp16_kvec[kid].spk_kptr)<<<grid_size, block_size, smem_size, stream>>>(SPK_KPARAM_LIST);
                    }

                    if (splitk > 1 || splitf > 1) {
                        int spk_width_v8  = num_flt_per_grp_pad * conv_param.num_grp / pad_size;
                        int spk_height_v1 = out_hw * conv_param.in_num;

                        dim3 merge_grid_size, merge_block_size;
                        merge_block_size.x = 64; // empirical value
                        merge_block_size.y = 1;
                        merge_block_size.z = 1;

                        merge_grid_size.x = spk_height_v1;
                        merge_grid_size.y = DivUp(spk_width_v8, merge_block_size.x);
                        merge_grid_size.z = 1;

                        MergeConvSplitResults<<<merge_grid_size, merge_block_size, 0, stream>>>(MERGE_KPARAM_LIST);
                    }
                }
            }

            cudaEventRecord(begin, stream);

            for (int i = 0; i < TIMES; i++) {
                if (g_fp16_kvec[kid].ktype == CONV_IDXN_C2 || g_fp16_kvec[kid].ktype == CONV_IDXN_C4 ||
                    g_fp16_kvec[kid].ktype == CONV_IDXN_C32) {
                    int tile_k_per_step = g_fp16_kvec[kid].tile_k_per_step;

                    int img_pad_size = pad_size;
                    int flt_pad_size = g_fp16_kvec[kid].flt_pad_size;
                    int out_nhw      = out_hw * conv_param.in_num;

                    int in_chl_per_grp_pad  = Align(num_chl_per_grp, img_pad_size);
                    int flt_chl_per_grp_pad = Align(num_chl_per_grp, flt_pad_size);
                    int num_flt_per_grp_pad = Align(num_flt_per_grp, img_pad_size);

                    int kloop_num    = DivUp(flt_hw * flt_chl_per_grp_pad, g_fp16_kvec[kid].tile_k_per_cta);
                    int koff_num_pad = Align(kloop_num * (g_fp16_kvec[kid].tile_k_per_cta / flt_pad_size), WARP_SIZE);

                    (g_fp16_kvec[kid].idx_kptr)<<<grid_size, block_size, 0, stream>>>(IDX_KPARAM_LIST);
                } else if (g_fp16_kvec[kid].ktype == CONV_2SPK_F1 || g_fp16_kvec[kid].ktype == CONV_2SPK_F3 ||
                           g_fp16_kvec[kid].ktype == CONV_2SPK_FN || g_fp16_kvec[kid].ktype == CONV_2SPK_FS ||
                           g_fp16_kvec[kid].ktype == CONV_SWZL_F1 || g_fp16_kvec[kid].ktype == CONV_SWZL_F3 ||
                           g_fp16_kvec[kid].ktype == CONV_SWZL_FN) {

                    int kloop_num = (flt_hw / splitf) * DivUp(num_chl_per_grp_pad, g_fp16_kvec[kid].tile_k_per_cta);

                    lut_t in_lut, flt_lut;
                    int in_lut_size, flt_lut_size;

                    InitializeInputLut(in_lut_size, in_lut.idx, conv_param.flt_height, conv_param.flt_width, conv_param.in_height, conv_param.in_width, conv_param.pad_height, conv_param.pad_width, conv_param.hole_height, conv_param.hole_width, num_chl_per_grp_pad, conv_param.num_grp, g_fp16_kvec[kid].tile_k_per_cta, pad_size);

                    InitializeFilterLut(flt_lut_size, flt_lut.idx, conv_param.flt_height, conv_param.flt_width, num_chl_per_grp_pad, g_fp16_kvec[kid].tile_k_per_cta, pad_size);

                    if (splitk == 1) {
                        g_fp16_kvec[kid].AdaptLutKernelSMemSize();

                        if(g_fp16_kvec[kid].ktype == CONV_SWZL_F1 || g_fp16_kvec[kid].ktype == CONV_SWZL_F3 || g_fp16_kvec[kid].ktype == CONV_SWZL_FN)
                            (g_fp16_kvec[kid].lut_kptr)<<<grid_size, block_size, smem_size, stream>>>(SWZL_LUT_KPARAM_LIST);
                        else {
                            (g_fp16_kvec[kid].lut_kptr)<<<grid_size, block_size, smem_size, stream>>>(LUT_KPARAM_LIST);
                        }
                    } else {
                        int num_chl_per_spk_head, num_chl_per_spk_tail;

                        InitializeNumChlPerSpk(num_chl_per_spk_head, num_chl_per_spk_tail, conv_param.num_chl, conv_param.num_grp, pad_size, g_fp16_kvec[kid].tile_k_per_cta, splitk);

                        g_fp16_kvec[kid].AdaptSpkKernelSMemSize();

                        if(g_fp16_kvec[kid].ktype == CONV_SWZL_F1 || g_fp16_kvec[kid].ktype == CONV_SWZL_F3 || g_fp16_kvec[kid].ktype == CONV_SWZL_FN)
                            (g_fp16_kvec[kid].spk_kptr)<<<grid_size, block_size, smem_size, stream>>>(SWZL_SPK_KPARAM_LIST);
                        else
                            (g_fp16_kvec[kid].spk_kptr)<<<grid_size, block_size, smem_size, stream>>>(SPK_KPARAM_LIST);
                    }

                    if (splitk > 1 || splitf > 1) {
                        int spk_width_v8  = num_flt_per_grp_pad * conv_param.num_grp / pad_size;
                        int spk_height_v1 = out_hw * conv_param.in_num;

                        dim3 merge_grid_size, merge_block_size;
                        merge_block_size.x = 64; // empirical value
                        merge_block_size.y = 1;
                        merge_block_size.z = 1;

                        merge_grid_size.x = spk_height_v1;
                        merge_grid_size.y = DivUp(spk_width_v8, merge_block_size.x);
                        merge_grid_size.z = 1;

                        MergeConvSplitResults<<<merge_grid_size, merge_block_size, 0, stream>>>(MERGE_KPARAM_LIST);
                    }
                }
            }

            cudaEventRecord(end, stream);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&elapsed, begin, end);

            if (elapsed < minTime) {
                algo_param.algo_type = "TuringHMMAImpgemm";
                algo_param.algo_name = g_fp16_kvec[kid].kname;
                algo_param.kid    = kid;
                algo_param.splitk = splitk;
                algo_param.splitf = splitf;
                minTime           = elapsed;
            }
        }
    }

    LOG(DEBUG) << "algo_type:" << algo_param.algo_type << 
                " algo_name:" << algo_param.algo_name << 
                " kid:" << algo_param.kid <<
                " splitk:" << algo_param.splitk <<
                " splitf:" << algo_param.splitf;
    LOG(DEBUG) << "minTime: " << minTime;

    if (is_out_grp_pad) {
        PPLCUDAConvolutionCvtOutput(stream, d_output, final_out, type, conv_param);
    }

    cudaEventDestroy(begin);
    cudaEventDestroy(end);

    g_conv_shape_hash[conv_shape_hash] = algo_param;
    return minTime;
#else
    return 0.0;
#endif
}

void PPLCUDAConvolutionForwardImp(
    const cudaDeviceProp& device_prop,
    cudaStream_t &stream,
    ppl::common::datatype_t type,
    int4 *d_input,
    int4 *d_flt,
    int4 *d_output,
    int4 *bias,
    int4 *d_temp_buf,
    algo_param_t &algo_param,
    conv_param_t &conv_param,
    fuse_param_t &fuse_param)
{
#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 9020
    if (algo_param.algo_type == "CutlassHConv") {
        int device_arch = device_prop.major * 10 + device_prop.minor;
        // nhwc
        const int n = conv_param.in_num;
        const int h = conv_param.in_height;
        const int w = conv_param.in_width;
        const int c = conv_param.num_chl_pad;
        // krsc
        const int k = conv_param.num_flt_pad;
        const int r = conv_param.flt_height;
        const int s = conv_param.flt_width;
        // npqk
        const int p = conv_param.out_height;
        const int q = conv_param.out_width;

        int pad_h = conv_param.pad_height;
        int pad_w = conv_param.pad_width;
        int stride_h = conv_param.stride_height;
        int stride_w = conv_param.stride_width;
        int dilation_h = conv_param.hole_height;
        int dilation_w = conv_param.hole_height;

        cutlass::library::ConvFunctionalKey functional_key(
            cutlass::library::Provider::kCUTLASS, cutlass::library::ConvKind::kFprop,
            cutlass::library::NumericTypeID::kF16, cutlass::library::LayoutTypeID::kTensorNHWC,
            cutlass::library::NumericTypeID::kF16, cutlass::library::LayoutTypeID::kTensorNHWC,
            cutlass::library::NumericTypeID::kF16, cutlass::library::LayoutTypeID::kTensorNHWC,
            cutlass::library::NumericTypeID::kF16, cutlass::library::NumericTypeID::kF16
        );

        void* x_ptr = d_input;
        void* w_ptr = d_flt;
        void* y_ptr = d_output;
        void* bias_ptr = nullptr;
        if (conv_param.has_bias == 1) bias_ptr = bias;

        cutlass::conv::Conv2dProblemSize problem_size(
            n, h, w, c, k, r, s, p, q, 
            pad_h, pad_w, stride_h, stride_w, 
            dilation_h, dilation_w,
            cutlass::conv::Mode::kCrossCorrelation 
        );
        cutlass::library::Conv2dConfiguration configuraion;
        configuraion.split_k_mode = cutlass::conv::SplitKMode::kSerial;
        configuraion.problem_size = problem_size;
        configuraion.stride_a = {c, w * c, h * w * c};
        configuraion.stride_b = {c, s * c, r * s * c};
        configuraion.stride_c = {0, 0, 0};
        cutlass::library::ConvArguments arguments;
        arguments.A = x_ptr;
        arguments.B = w_ptr;
        arguments.reordered_B = nullptr;
        arguments.C = bias_ptr;
        arguments.D = y_ptr;

        union SP {
            float f{};
            half h;
        };

        SP alpha;
        SP beta;
        alpha.h = static_cast<half>(1.0F);
        if (conv_param.has_bias != 1) beta.h = static_cast<half>(0.0F);
        else beta.h = static_cast<half>(1.0F);
    
        arguments.alpha = &alpha;
        arguments.beta = &beta;
        arguments.pointer_mode = cutlass::library::ScalarPointerMode::kHost;
        
        // const cutlass::library::Operation* fastest_operation = nullptr;
        Conv2dOperationCacheKey cache_key(functional_key, configuraion, arguments);
        const cutlass::library::Operation* fastest_operation;
        int dev = 0;
        const auto& device_cache = cache[dev];
        const auto& it = device_cache.find(cache_key);
        if (it != device_cache.end()) {
            fastest_operation = it->second;
            const size_t host_workspace_size = fastest_operation->get_host_workspace_size(&configuraion);
            const size_t device_workspace_size = fastest_operation->get_device_workspace_size(&configuraion);
            std::vector<uint8_t> host_workspace(host_workspace_size, 0);
            auto init_status = fastest_operation->initialize(&configuraion, host_workspace.data(),
                                                    d_temp_buf, stream);
            if (init_status != cutlass::Status::kSuccess) LOG(ERROR) << "init_status != cutlass::Status::kSuccess";
            auto run_status = fastest_operation->run(&arguments, host_workspace.data(), d_temp_buf,
                                            stream);
            if (run_status != cutlass::Status::kSuccess) LOG(ERROR) << "run_status != cutlass::Status::kSuccess";
            CUDA_CHECK(cudaGetLastError());
            return;
        } else {
            const auto& operations_map_it =
                cutlass::library::Singleton::get().operation_table.conv2d_operations.find(functional_key);
            if (operations_map_it == cutlass::library::Singleton::get().operation_table.conv2d_operations.cend())
                LOG(ERROR) << "Can't find the operations_map_it";
            const cutlass::library::ConvOperationVectorMap& operations_map = operations_map_it->second;
            for (const auto& pair : operations_map) {
                for (auto operation : pair.second) {
                    if (algo_param.algo_name != operation->description().name) continue;
                    if (operation->description().tile_description.minimum_compute_capability > device_arch
                        || operation->description().tile_description.maximum_compute_capability < device_arch) {
                        continue;
                    }
                    auto status = operation->can_implement(&configuraion, &arguments);
                    if (status != cutlass::Status::kSuccess) continue;
                    const size_t host_workspace_size = operation->get_host_workspace_size(&configuraion);
                    const size_t device_workspace_size = operation->get_device_workspace_size(&configuraion);
                    if (device_workspace_size > algo_param.splitk) continue;
                    std::vector<uint8_t> host_workspace(host_workspace_size, 0);
                    if (operation->initialize(&configuraion, host_workspace.data(), d_temp_buf, stream)
                            != cutlass::Status::kSuccess) {
                        continue;   // Werid: there are several operation has the same name?
                    }
                    if (operation == nullptr) LOG(ERROR) << "operation == nullptr";
                    auto run_status = operation->run(&arguments, host_workspace.data(), d_temp_buf,
                                                    stream);
                    if (run_status != cutlass::Status::kSuccess) LOG(ERROR) << "run_status != cutlass::Status::kSuccess";
                    CUDA_CHECK(cudaGetLastError());
                    cache[dev][cache_key] = operation;
                    return;
                }
            }
        }
    }

    if (!is_g_fp16_kvec_initialized)
        InitializeFP16ConvKernelContainer(g_fp16_kvec, device_prop, type);

    unsigned int kid    = algo_param.kid;
    unsigned int splitk = algo_param.splitk;
    unsigned int splitf = algo_param.splitf;

    int pad_size = GetPadSize(type);

    int num_chl_per_grp = conv_param.num_chl / conv_param.num_grp;
    int num_flt_per_grp = conv_param.num_flt / conv_param.num_grp;

    int num_chl_per_grp_pad = Align(num_chl_per_grp, pad_size);
    int num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size);

    int in_hw  = conv_param.in_height * conv_param.in_width;
    int flt_hw = conv_param.flt_height * conv_param.flt_width;
    int out_hw = conv_param.out_height * conv_param.out_width;

    int concat_offset_v8 = fuse_param.concat_offset / pad_size;
    int concat_stride_v8 = DivUp(fuse_param.concat_stride, pad_size); // should use DivUp

    bool is_in_grp_pad  = num_chl_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;
    bool is_out_grp_pad = num_flt_per_grp_pad != num_flt_per_grp && conv_param.num_grp != 1;

    uint64_t buf_off_v4 = 0;

    int4 *pad_input  = d_input;
    int4 *pad_output = d_output;

    if (is_in_grp_pad) {
        pad_input = d_temp_buf;
        buf_off_v4 += GetCvtInputSize(type, conv_param, num_chl_per_grp_pad) / (_4INT_TO_INT4_ * _INT_TO_4BYTE_);

        PPLCUDAConvolutionCvtInput(stream, pad_input, d_input, type, conv_param);
    }

    if (is_out_grp_pad) {
        pad_output = d_temp_buf + buf_off_v4;
        buf_off_v4 += getCvtOutputSize(type, conv_param, num_flt_per_grp_pad) / (_4INT_TO_INT4_ * _INT_TO_4BYTE_);
    }

    int4 *final_out = fuse_param.has_concat ? (int4 *)fuse_param.post_concat : pad_output;

    int4 *splitk_buf = d_temp_buf + buf_off_v4;
    int4 *conv_out   = (splitk > 1 || splitf > 1) ? splitk_buf : final_out;

    __half2 clip_min     = __float2half2_rn(fuse_param.clip_min);
    __half2 clip_max     = __float2half2_rn(fuse_param.clip_max);
    __half2 elt_clip_min = __float2half2_rn(fuse_param.elt_clip_min);
    __half2 elt_clip_max = __float2half2_rn(fuse_param.elt_clip_max);
    __half leaky         = __float2half(fuse_param.leaky);
    __half elt_leaky     = __float2half(fuse_param.elt_leaky);

    dim3 block_size, grid_size;

    block_size.x = g_fp16_kvec[kid].cta_size_in_thd;
    block_size.y = 1;
    block_size.z = 1;

    int smem_size = g_fp16_kvec[kid].smem_size;

    if(g_fp16_kvec[kid].ktype == CONV_SWZL_F1 || g_fp16_kvec[kid].ktype == CONV_SWZL_F3 || \
            g_fp16_kvec[kid].ktype == CONV_SWZL_FN) {
        grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, g_fp16_kvec[kid].tile_n_per_cta);
        grid_size.y = DivUp(num_flt_per_grp_pad, g_fp16_kvec[kid].tile_m_per_cta);
    } else {
        grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, g_fp16_kvec[kid].tile_m_per_cta);
        grid_size.y = DivUp(num_flt_per_grp_pad, g_fp16_kvec[kid].tile_n_per_cta);
    }

    grid_size.z = conv_param.num_grp * splitk * splitf;

    if (g_fp16_kvec[kid].ktype == CONV_IDXN_C2 || g_fp16_kvec[kid].ktype == CONV_IDXN_C4 ||
        g_fp16_kvec[kid].ktype == CONV_IDXN_C32) {
        int img_pad_size = pad_size;
        int flt_pad_size = g_fp16_kvec[kid].flt_pad_size;

        int out_nhw = out_hw * conv_param.in_num;

        int in_chl_per_grp_pad  = Align(num_chl_per_grp, img_pad_size);
        int flt_chl_per_grp_pad = Align(num_chl_per_grp, flt_pad_size);
        int num_flt_per_grp_pad = Align(num_flt_per_grp, img_pad_size);

        int kloop_num    = DivUp(flt_hw * flt_chl_per_grp_pad, g_fp16_kvec[kid].tile_k_per_cta);
        int koff_num_pad = Align(kloop_num * (g_fp16_kvec[kid].tile_k_per_cta / flt_pad_size), WARP_SIZE);

        (g_fp16_kvec[kid].idx_kptr)<<<grid_size, block_size, 0, stream>>>(IDX_KPARAM_LIST);

    } else if (g_fp16_kvec[kid].ktype == CONV_2SPK_F1 || g_fp16_kvec[kid].ktype == CONV_2SPK_F3 ||
               g_fp16_kvec[kid].ktype == CONV_2SPK_FN || g_fp16_kvec[kid].ktype == CONV_2SPK_FS ||
               g_fp16_kvec[kid].ktype == CONV_SWZL_F1 || g_fp16_kvec[kid].ktype == CONV_SWZL_F3 ||
               g_fp16_kvec[kid].ktype == CONV_SWZL_FN) {

        int kloop_num = (flt_hw / splitf) * DivUp(num_chl_per_grp_pad, g_fp16_kvec[kid].tile_k_per_cta);

        lut_t in_lut, flt_lut;
        int in_lut_size, flt_lut_size;

        InitializeInputLut(in_lut_size, in_lut.idx, conv_param.flt_height, conv_param.flt_width, conv_param.in_height, conv_param.in_width, conv_param.pad_height, conv_param.pad_width, conv_param.hole_height, conv_param.hole_width, num_chl_per_grp_pad, conv_param.num_grp, g_fp16_kvec[kid].tile_k_per_cta, pad_size);

        InitializeFilterLut(flt_lut_size, flt_lut.idx, conv_param.flt_height, conv_param.flt_width, num_chl_per_grp_pad, g_fp16_kvec[kid].tile_k_per_cta, pad_size);

        if (splitk == 1) {
            g_fp16_kvec[kid].AdaptLutKernelSMemSize();

            if(g_fp16_kvec[kid].ktype == CONV_SWZL_F1 || g_fp16_kvec[kid].ktype == CONV_SWZL_F3 || g_fp16_kvec[kid].ktype == CONV_SWZL_FN)
                (g_fp16_kvec[kid].lut_kptr)<<<grid_size, block_size, smem_size, stream>>>(SWZL_LUT_KPARAM_LIST);
            else {
                (g_fp16_kvec[kid].lut_kptr)<<<grid_size, block_size, smem_size, stream>>>(LUT_KPARAM_LIST);
            }

        } else {
            int num_chl_per_spk_head, num_chl_per_spk_tail;

            InitializeNumChlPerSpk(num_chl_per_spk_head, num_chl_per_spk_tail, conv_param.num_chl, conv_param.num_grp, pad_size, g_fp16_kvec[kid].tile_k_per_cta, splitk);

            g_fp16_kvec[kid].AdaptSpkKernelSMemSize();

            if(g_fp16_kvec[kid].ktype == CONV_SWZL_F1 || g_fp16_kvec[kid].ktype == CONV_SWZL_F3 || g_fp16_kvec[kid].ktype == CONV_SWZL_FN)
                (g_fp16_kvec[kid].spk_kptr)<<<grid_size, block_size, smem_size, stream>>>(SWZL_SPK_KPARAM_LIST);
            else
                (g_fp16_kvec[kid].spk_kptr)<<<grid_size, block_size, smem_size, stream>>>(SPK_KPARAM_LIST);
        }
    }

    if (splitk > 1 || splitf > 1) {
        int spk_width_v8  = num_flt_per_grp_pad * conv_param.num_grp / pad_size;
        int spk_height_v1 = out_hw * conv_param.in_num;

        dim3 merge_grid_size, merge_block_size;
        merge_block_size.x = 64;
        merge_block_size.y = 1;
        merge_block_size.z = 1;

        merge_grid_size.x = spk_height_v1;
        merge_grid_size.y = DivUp(spk_width_v8, merge_block_size.x);
        merge_grid_size.z = 1;

        MergeConvSplitResults<<<merge_grid_size, merge_block_size, 0, stream>>>(MERGE_KPARAM_LIST);
    }

    if (is_out_grp_pad) {
        PPLCUDAConvolutionCvtOutput(stream, d_output, final_out, type, conv_param);
    }
#endif
}

void kernel_info_t::AdaptLutKernelSMemSize()
{
    if(smem_size <= MAX_STATIC_SMEM_SIZE_PER_CTA)
        return;

    cudaFuncSetAttribute(lut_kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    return;
}

void kernel_info_t::AdaptSpkKernelSMemSize()
{
    if(smem_size <= MAX_STATIC_SMEM_SIZE_PER_CTA)
        return;

    cudaFuncSetAttribute(spk_kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    return;
}

void kernel_info_t::AdaptInt8LutKernelSMemSize()
{
    if(smem_size <= MAX_STATIC_SMEM_SIZE_PER_CTA)
        return;

    cudaFuncSetAttribute(int8_lut_kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    return;
}

void kernel_info_t::AdaptInt8SpkKernelSMemSize()
{
    if(smem_size <= MAX_STATIC_SMEM_SIZE_PER_CTA)
        return;

    cudaFuncSetAttribute(int8_spk_kptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    return;
}

/* -----------------  JIT FP16 KERNEL ------------------ */

ppl::common::RetCode algo_param_t::BuildAlgoName()
{
    std::string algo_name("nv");

    if (this->conv_type == "idxn")
        algo_name += "Idxn";
    else if (this->conv_type == "2spk")
        algo_name += "2spk";
    else if (this->conv_type == "swzl")
        algo_name += "Swzl";

    if (this->mma_shape == "hmma1688" && this->tiles.buf <= 2)
        algo_name += "Sm75Fp16Conv_hmma1688_nhwc_";
    else if (this->mma_shape == "hmma1688" && this->tiles.buf > 2)
        algo_name += "Sm80Fp16Conv_hmma1688_nhwc_";
    else if (this->mma_shape == "hmma16816")
        algo_name += "Sm80Fp16Conv_hmma16816_nhwc_";
    else if (this->mma_shape == "imma8816" && this->tiles.buf <= 2)
        algo_name += "Sm75Int8Conv_imma8816_nhwc_";
    else if (this->mma_shape == "imma8816" && this->tiles.buf > 2)
        algo_name += "Sm80Int8Conv_imma8816_nhwc_";
    else if (this->mma_shape == "imma16816")
        algo_name += "Sm80Int8Conv_imma16816_nhwc_";
    else if (this->mma_shape == "imma16832")
        algo_name += "Sm80Int8Conv_imma16832_nhwc_";

    if (this->conv_type == "idxn") {
        algo_name += "b" + std::to_string(this->tiles.m_cta)  + "x" + std::to_string(this->tiles.n_cta)  + "_" + \
                     "w" + std::to_string(this->tiles.m_warp) + "x" + std::to_string(this->tiles.n_warp) + "_" + \
                     "k" + std::to_string(this->tiles.k_cta)  + "_" + \
                     "s" + std::to_string(this->tiles.k_per_step);
    } else if (this->conv_type == "2spk") {
        if (this->tiles.flt_size == 1)
            algo_name += "f1_";
        else if(this->tiles.flt_size == 3)
            algo_name += "f3_";
        else if(this->tiles.flt_size == 0)
            algo_name += "fn_";
        else if(this->tiles.flt_size == 11)
            algo_name += "fs_";

        algo_name += "b" + std::to_string(this->tiles.m_cta)  + "x" + std::to_string(this->tiles.n_cta)  + "_" + \
                     "w" + std::to_string(this->tiles.m_warp) + "x" + std::to_string(this->tiles.n_warp) + "_" + \
                     "k" + std::to_string(this->tiles.k_cta)  + "_" + \
                     "s" + std::to_string(this->tiles.k_per_set)  + "_" + \
                     "buf" + std::to_string(this->tiles.buf);
        
        if (this->splitk > 1)
            algo_name += "_spk" + std::to_string(this->splitk);
    } else if (this->conv_type == "swzl") {
        if (this->tiles.flt_size == 1)
            algo_name += "f1_";
        else if(this->tiles.flt_size == 3)
            algo_name += "f3_";
        else if(this->tiles.flt_size == 0)
            algo_name += "fn_";

        algo_name += "b" + std::to_string(this->tiles.m_cta)  + "x" + std::to_string(this->tiles.n_cta)  + "_" + \
                     "w" + std::to_string(this->tiles.m_warp) + "x" + std::to_string(this->tiles.n_warp) + "_" + \
                     "k" + std::to_string(this->tiles.k_cta)  + "_" + \
                     "buf" + std::to_string(this->tiles.buf);

        if (this->splitk > 1)
            algo_name += "_spk" + std::to_string(this->splitk);
    }

    this->algo_name = algo_name;

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode algo_param_t::ParseAlgoName()
{
#ifdef PPLNN_ENABLE_CUDA_JIT
    std::stringstream algo_name_str(this->algo_name);
    std::vector<std::string> algo_name_substrs;
    std::string substr;

    while(std::getline(algo_name_str, substr, '_')) {
       algo_name_substrs.push_back(substr);
    }

    this->mma_shape = algo_name_substrs[1];

    int m_mma = 0;
    int n_mma = 0;
    int type_size = 0;
    ppl::common::datatype_t type = ppl::common::DATATYPE_UNKNOWN;

    if (strstr(algo_name_substrs[0].c_str(), "Int8")) {
        type_size = 1;
        type =  ppl::common::DATATYPE_INT8;
    } else if (strstr(algo_name_substrs[0].c_str(), "Fp16")) {
        type_size = 2;
        type =  ppl::common::DATATYPE_FLOAT16;
    }

    if ( strstr(algo_name_substrs[0].c_str(), "Idxn") ) {
        this->conv_type = "idxn";

        sscanf(algo_name_substrs[3].c_str(), "b%dx%d", &(this->tiles.m_cta), &(this->tiles.n_cta));
        sscanf(algo_name_substrs[4].c_str(), "w%dx%d", &(this->tiles.m_warp), &(this->tiles.n_warp));
        sscanf(algo_name_substrs[5].c_str(), "k%d",    &(this->tiles.k_cta));
        sscanf(algo_name_substrs[6].c_str(), "s%d",    &(this->tiles.k_per_step));

        this->tiles.cta_size_in_thd = (this->tiles.m_cta / this->tiles.m_warp) *
                                      (this->tiles.n_cta / this->tiles.n_warp) *
                                      WARP_SIZE;

        this->tiles.flt_pad_size    = this->tiles.k_per_step / 4;

    } else if ( strstr(algo_name_substrs[0].c_str(), "2spk") ) {
        this->conv_type = "2spk";

        if(this->mma_shape == "hmma16816" || this->mma_shape == "hmma1688" || this->mma_shape == "imma16816" || \
                this->mma_shape == "imma16832") {
            m_mma = 16;
            n_mma = 8;
        } else if (this->mma_shape == "imma8816") {
            m_mma = 8;
            n_mma = 8;
        }
        if (strstr(algo_name_substrs[3].c_str(), "f1"))
            this->tiles.flt_size = 1;
        else if (strstr(algo_name_substrs[3].c_str(), "f3"))
            this->tiles.flt_size = 3;
        else if (strstr(algo_name_substrs[3].c_str(), "fn"))
            this->tiles.flt_size = 0;
        else if (strstr(algo_name_substrs[3].c_str(), "fs"))
            this->tiles.flt_size = 11;

        sscanf(algo_name_substrs[4].c_str(), "b%dx%d", &(this->tiles.m_cta), &(this->tiles.n_cta));
        sscanf(algo_name_substrs[5].c_str(), "w%dx%d", &(this->tiles.m_warp), &(this->tiles.n_warp));
        sscanf(algo_name_substrs[6].c_str(), "k%d",    &(this->tiles.k_cta));
        sscanf(algo_name_substrs[7].c_str(), "s%d",    &(this->tiles.k_per_set));
        sscanf(algo_name_substrs[8].c_str(), "buf%d",  &(this->tiles.buf));
        if(algo_name_substrs.size() == 10)
            sscanf(algo_name_substrs[9].c_str(), "spk%d",  &(this->splitk));

        this->tiles.cta_size_in_thd = (this->tiles.m_cta / this->tiles.m_warp) *
                                      (this->tiles.n_cta / this->tiles.n_warp) *
                                      (this->tiles.k_cta / this->tiles.k_per_set) *
                                      WARP_SIZE;

        this->tiles.smem_size = Get2spkSmemUsage(type, type_size, this->tiles.m_cta, this->tiles.n_cta, \
                this->tiles.k_cta, (this->tiles.k_cta / this->tiles.k_per_set), this->tiles.buf);

    } else if ( strstr(algo_name_substrs[0].c_str(), "Swzl") ) {
        this->conv_type = "swzl";

        if(this->mma_shape == "hmma16816" || this->mma_shape == "hmma1688" || this->mma_shape == "imma16816" || \
                this->mma_shape == "imma16832") {
            m_mma = 8;
            n_mma = 16;
        } else if (this->mma_shape == "imma8816") {
            m_mma = 8;
            n_mma = 8;
        }

        if (strstr(algo_name_substrs[3].c_str(), "f1"))
            this->tiles.flt_size = 1;
        else if (strstr(algo_name_substrs[3].c_str(), "f3"))
            this->tiles.flt_size = 3;
        else if (strstr(algo_name_substrs[3].c_str(), "fn"))
            this->tiles.flt_size = 0;

        sscanf(algo_name_substrs[4].c_str(), "b%dx%d", &(this->tiles.m_cta), &(this->tiles.n_cta));
        sscanf(algo_name_substrs[5].c_str(), "w%dx%d", &(this->tiles.m_warp), &(this->tiles.n_warp));
        sscanf(algo_name_substrs[6].c_str(), "k%d",    &(this->tiles.k_cta));
        sscanf(algo_name_substrs[7].c_str(), "buf%d",  &(this->tiles.buf));
        if(algo_name_substrs.size() == 9)
            sscanf(algo_name_substrs[8].c_str(), "spk%d",  &(this->splitk));

        this->tiles.cta_size_in_thd = (this->tiles.m_cta / this->tiles.m_warp) *
                                      (this->tiles.n_cta / this->tiles.n_warp) *
                                      WARP_SIZE;

        this->tiles.smem_size = GetSwzlSmemUsage(type, type_size, this->tiles.m_cta, this->tiles.n_cta, \
                this->tiles.k_cta, this->tiles.m_warp, this->tiles.n_warp, m_mma, n_mma, \
                this->tiles.buf, this->tiles.cta_size_in_thd / WARP_SIZE);

    } else {
        return ppl::common::RC_NOT_FOUND;
    }
#endif
    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode GetFp16ConvKernelNominees(
    const cudaDeviceProp& device_prop,
    ppl::common::datatype_t type,
    conv_param_t &conv_param,
    std::vector<std::string> & knames,
    std::vector<algo_param_t> & params,
    std::string & sources,
    bool spk_only)
{
#ifdef PPLNN_ENABLE_CUDA_JIT
    int pad_size            = GetPadSize(type);
    int num_grp             = conv_param.num_grp;
    int num_chl_per_grp     = conv_param.num_chl / num_grp;
    int num_flt_per_grp     = conv_param.num_flt / num_grp;
    int num_chl_per_grp_pad = Align(num_chl_per_grp, pad_size);
    // int num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size);

    int batch               = conv_param.in_num;
    int flt_h               = conv_param.flt_height;
    int flt_w               = conv_param.flt_width;
    int flt_hw              = flt_h * flt_w;
    // int in_hw               = conv_param.in_height  * conv_param.in_width;
    int out_w               = conv_param.out_width;
    int out_hw              = conv_param.out_height * conv_param.out_width;

    int type_size = ppl::common::GetSizeOfDataType(type);

    int m_conv = Align(batch * out_hw,  pad_size);
    int n_conv = Align(num_flt_per_grp, pad_size);

    int sm_num = device_prop.multiProcessorCount;
    int device_arch      = device_prop.major * 10 + device_prop.minor;
    if (device_arch < 75) {
        LOG(ERROR)<<"pplnn should be compiled on cuda >= 10.2 and run on architeture >= sm_75";
        return ppl::common::RC_UNSUPPORTED;
    }
    int max_regs_per_thd = 255;
    int max_regs_per_sm  = device_prop.regsPerMultiprocessor;
#if __CUDACC_VER_MAJOR__ >= 11
    int max_ctas_per_sm  = device_prop.maxBlocksPerMultiProcessor;
#else
    int max_ctas_per_sm = 32;
#endif
    int max_thds_per_sm  = device_prop.maxThreadsPerMultiProcessor;
    int max_smem_per_sm  = device_prop.sharedMemPerMultiprocessor;
    int max_regs_per_cta = device_prop.regsPerBlock;
    int max_smem_per_cta = device_prop.sharedMemPerBlock;
    int max_dyn_smem_per_cta = 0;

    std::string mma_shape = "";
    int splitk = 1;
    int splitf = 1;
    int m_mma = 0;
    int n_mma = 0;
    int k_mma = 0;
    int m_mma_max = 0;
    int n_mma_max = 0;
    int k_mma_max = 0;
    int k_blk_mma = 0;
    int buf_num_max = 0;

    int cpi_mma = 0;
    int cpi_ldg32_l1d = 0;
    int cpi_ldg64_l1d = 0;
    int cpi_ldg128_l1d = 0;
    int cpi_ldg32_l2 = 0;
    int cpi_ldg64_l2 = 0;
    int cpi_ldg128_l2 = 0;
    int cpi_lds32 = 0;
    int cpi_lds64 = 0;
    int cpi_lds128 = 0;
    int cpi_sts32 = 0;
    int cpi_sts64 = 0;
    int cpi_sts128 = 0;
    int latency_mma = 0;
    int latency_l2_cache = 0;
    int latency_dram = 0;

    std::vector<std::pair<algo_param_t, float>> nominees;
    algo_param_t nominee;
    nominee.algo_type = "TuringHMMAImpgemm";

    GetHardwareInfo(device_arch, type, num_chl_per_grp, cpi_mma, latency_mma, cpi_ldg32_l1d, cpi_ldg64_l1d, \
            cpi_ldg128_l1d, cpi_ldg32_l2, cpi_ldg64_l2, cpi_ldg128_l2, cpi_lds32, cpi_lds64, cpi_lds128, \
            cpi_sts32, cpi_sts64, cpi_sts128, latency_l2_cache, latency_dram, max_dyn_smem_per_cta);

    if (num_chl_per_grp <= 32 && !spk_only) {
        int k_per_step = 0;
        if (num_chl_per_grp > 0 && num_chl_per_grp <= 2) {
            k_per_step = 8;
        } else if (num_chl_per_grp > 2 && num_chl_per_grp <= 4) {
            k_per_step = 16;
        } else if (num_chl_per_grp > 4 && num_chl_per_grp <= 32) {
            k_per_step = 32;
        }
        
        GetIdxnMmaInfo(device_arch, type, num_chl_per_grp, mma_shape, m_mma, n_mma, k_mma, m_mma_max, n_mma_max, k_mma_max);

        int flt_pad_size = k_per_step >> 2;
        int flt_chl_per_grp_pad = Align(num_chl_per_grp, flt_pad_size);
        int k_conv = flt_hw * flt_chl_per_grp_pad;

        for(int k_num = 1; k_num <= 2; k_num *= 2)
            for(int m_warp = m_mma; m_warp <= m_mma_max; m_warp *= 2)
                for(int n_warp = n_mma; n_warp <= n_mma_max; n_warp *= 2)
                    for(int m_warp_num = 1; m_warp_num <= 4; m_warp_num *= 2)
                        for(int n_warp_num = 1; n_warp_num <= 4; n_warp_num *= 2) {

                            int m_cta = m_warp * m_warp_num;
                            int n_cta = n_warp * n_warp_num;
                            int k_cta = k_per_step * k_num;
                            int cta_size_in_warp = m_warp_num * n_warp_num;
                            int cta_size_in_thd  = cta_size_in_warp * WARP_SIZE;

                            int kloop_total  = DivUp(flt_hw * flt_chl_per_grp_pad, k_cta);
                            int kloop_num = kloop_total;

                            int m_cta_num = DivUp(m_conv, m_cta);
                            int n_cta_num = DivUp(n_conv, n_cta);
                            int cta_num = m_cta_num * n_cta_num;

                            if( m_warp_num == 4 && n_warp_num == 4 ) continue;
                            if(m_warp == m_mma_max && n_warp == n_mma_max) continue;

                            int kloop_time = DivUp(kloop_num * (k_cta / flt_pad_size), cta_size_in_thd);
                            if(kloop_time != 1) continue;

                            int regs_per_thd = GetIdxnRegsPerThread(type, m_cta, n_cta, m_warp, n_warp, k_per_step, m_mma, n_mma, k_mma, cta_size_in_thd);
                            int regs_per_cta = regs_per_thd * cta_size_in_thd;
                            if (regs_per_thd > max_regs_per_thd) continue;
                            if (regs_per_cta > max_regs_per_cta) continue; 

                            int smem_per_cta = GetIdxnSmemUsage(m_cta, cta_size_in_thd);
                            if (smem_per_cta > max_smem_per_cta) continue;

                            float eff_score = GetEfficiencyScore(m_cta, n_cta, k_cta, kloop_total, m_conv, n_conv, k_conv);
                            if(eff_score < 0.5) continue;

                            float cta_launch_times = 0.f;
                            float occ_score = GetOccupancyScore(cta_size_in_thd, cta_size_in_warp, sm_num, cta_num, regs_per_cta, smem_per_cta, \
                                    max_ctas_per_sm, max_thds_per_sm, max_regs_per_sm, max_smem_per_sm, cta_launch_times);
                            if(occ_score < 0.5) continue;

                            float pip_score = GetIdxnPipelineScore(type_size, cta_launch_times, out_w, cta_size_in_thd, cta_size_in_warp, m_cta, n_cta, k_cta, m_warp, n_warp, \
                                    k_per_step, m_mma, n_mma, k_mma, cpi_mma, cpi_ldg32_l1d, cpi_ldg64_l1d, cpi_ldg128_l1d, cpi_ldg32_l2, \
                                    cpi_ldg64_l2, cpi_ldg128_l2, latency_mma, latency_l2_cache, latency_dram);

                            float score = eff_score + occ_score + pip_score;
                            nominee.SetIdxnKernelParam(m_cta, n_cta, k_cta, m_warp, n_warp, k_per_step, flt_pad_size, cta_size_in_thd, smem_per_cta, splitk, splitf, mma_shape);

                            nominees.push_back(std::make_pair(nominee, score));
                        }

        if(nominees.size() == 0) {
            // nvIdxnConv_b128x64_w64x32
            nominee.SetIdxnKernelParam(128, 64, k_per_step, 64, 32, k_per_step, flt_pad_size, 128, 4096, 1, 1, mma_shape);
            nominees.push_back(std::make_pair(nominee, 0.f));
        }

    } else {
        int flt_size = 0;
        int k_conv = flt_hw * num_chl_per_grp_pad;
        int estimate_cta_num = GetEstimateCtaNumber(m_conv, n_conv, num_grp);

        if(conv_param.flt_height == 1 && conv_param.flt_width == 1)
            flt_size = 1;
        else if(conv_param.flt_height == 3 && conv_param.flt_width == 3)
            flt_size = 3;
        else
            flt_size = 0;

        if(estimate_cta_num <= sm_num || spk_only) {

            Get2spkMmaInfo(device_arch, type, mma_shape, m_mma, n_mma, k_mma, m_mma_max, n_mma_max, k_mma_max, k_blk_mma, buf_num_max);
            
            const int SPLITK_OPTIONS[] = {1, 2, 4, 8};

            for(int spf = 0; spf < 2; spf++)
                for(int spk = 0; spk < 4; spk++)
                    for(int buf_num = 1; buf_num <= buf_num_max; buf_num++)
                        for(int k_per_set = k_mma; k_per_set <= k_mma_max; k_per_set *= 2)
                            for(int set_num = 1; set_num <= 4; set_num *= 2)
                                for(int m_warp = m_mma; m_warp <= m_mma_max; m_warp *= 2)
                                    for(int n_warp = n_mma; n_warp <= n_mma_max; n_warp *= 2)
                                        for(int m_warp_num = 1; m_warp_num <= 4; m_warp_num *= 2)
                                            for(int n_warp_num = 1; n_warp_num <= 4; n_warp_num *= 2) {

                                                int m_cta = m_warp * m_warp_num;
                                                int n_cta = n_warp * n_warp_num;
                                                int k_cta = k_per_set * set_num;
                                                int set_size_in_warp = m_warp_num * n_warp_num;
                                                int cta_size_in_warp = m_warp_num * n_warp_num * set_num;
                                                int set_size_in_thd  = set_size_in_warp * WARP_SIZE;
                                                int cta_size_in_thd  = cta_size_in_warp * WARP_SIZE;

                                                if( n_conv >= 64 && n_cta < 64 ) continue;

                                                splitk = SPLITK_OPTIONS[spk];
                                                if( splitk > 1 && splitk * k_cta >= Align(num_chl_per_grp, k_cta) ) continue;

                                                if(spf == 1) { splitf = flt_hw; flt_size = 11; }
                                                if(spf == 1 && splitf == 1) continue;
                                                if(splitk * splitf >= MAX_SPLIT_SIZE) continue;

                                                int m_cta_num = DivUp(m_conv, m_cta);
                                                int n_cta_num = DivUp(n_conv, n_cta);
                                                int cta_num = m_cta_num * n_cta_num * num_grp * splitk * splitf;

                                                int split = splitk * splitf;
                                                int kloop_total = flt_hw * DivUp(num_chl_per_grp_pad, k_cta);
                                                int kloop_num = kloop_total / split;

                                                if( k_cta != GetTileKSize(num_chl_per_grp_pad, kloop_num) ) continue;

                                                if( m_warp_num == 4 && n_warp_num == 4 ) continue;
                                                if(m_warp == m_mma_max && n_warp == n_mma_max) continue;
                                                if(cta_size_in_thd == 32 && k_cta == 64) continue;
                                                if(cta_size_in_thd <= 64 && k_cta == 128) continue;
                                                if(cta_size_in_thd <= 128 && k_cta == 256) continue;
                                                if(buf_num > kloop_num) continue;

                                                int regs_per_thd = Get2spkRegsPerThread(type, type_size, m_cta, n_cta, k_cta, m_warp, n_warp, k_per_set, \
                                                        m_mma, n_mma, k_mma, k_blk_mma, buf_num, cta_size_in_thd, set_size_in_thd);
                                                int regs_per_cta = regs_per_thd * cta_size_in_thd;
                                                if (regs_per_thd > max_regs_per_thd) continue;
                                                if (regs_per_cta > max_regs_per_cta) continue;

                                                int smem_per_cta = Get2spkSmemUsage(type, type_size, m_cta, n_cta, k_cta, set_num, buf_num);
                                                if (smem_per_cta > max_dyn_smem_per_cta) continue;

                                                float eff_score = GetEfficiencyScore(m_cta, n_cta, k_cta, kloop_total, m_conv, n_conv, k_conv);
                                                if(eff_score < 0.5) continue;

                                                float cta_launch_times = 0.f;
                                                float occ_score = GetOccupancyScore(cta_size_in_thd, cta_size_in_warp, \
                                                        sm_num, cta_num, regs_per_cta, smem_per_cta,  max_ctas_per_sm, \
                                                        max_thds_per_sm, max_regs_per_sm, max_smem_per_sm, cta_launch_times);
                                                if(occ_score < 0.5) continue;

                                                if( cta_launch_times > 1 ) continue;

                                                float pip_score = Get2spkPipelineScore(type_size, cta_launch_times, m_conv, n_conv, k_conv, \
                                                        kloop_num, splitk, splitf, out_w, cta_size_in_thd, cta_size_in_warp, sm_num, m_cta, \
                                                        n_cta, k_cta, m_warp, n_warp, k_per_set, set_num, buf_num, m_mma, n_mma, k_mma, k_mma_max, \
                                                        cpi_mma, cpi_ldg128_l1d, cpi_ldg128_l2, cpi_lds128, cpi_sts32, latency_mma, \
                                                        latency_l2_cache, latency_dram);

                                                float score = eff_score + occ_score + pip_score;
                                                nominee.Set2spkKernelParam(m_cta, n_cta, k_cta, m_warp, n_warp, k_per_set, \
                                                        flt_size, buf_num, cta_size_in_thd, smem_per_cta, splitk, splitf, mma_shape);

                                                nominees.push_back(std::make_pair(nominee, score));
                                            }

            if(nominees.size() == 0) {
                if(conv_param.flt_height == 1 && conv_param.flt_width == 1)
                    flt_size = 1;
                else if(conv_param.flt_height == 3 && conv_param.flt_width == 3)
                    flt_size = 3;
                else
                    flt_size = 0;

                // nv2spkConv_b64x64_w32x32_k64_s32_buf1
                nominee.Set2spkKernelParam(64, 64, 64, 32, 32, 32, flt_size, 1, 256, 16384, 1, 1, mma_shape);
                nominees.push_back(std::make_pair(nominee, 0.f));
            }

        } else {
            GetSwzlMmaInfo(device_arch, type, mma_shape, m_mma, n_mma, k_mma, m_mma_max, n_mma_max, k_mma_max, k_blk_mma, buf_num_max);

            int tmp = m_conv; m_conv = n_conv; n_conv = tmp;
            
            for(int buf_num = 1; buf_num <= buf_num_max; buf_num++)
                for(int k_cta = k_mma; k_cta <= k_mma_max; k_cta *= 2)
                    for(int m_warp = m_mma; m_warp <= m_mma_max; m_warp *= 2)
                        for(int n_warp = n_mma; n_warp <= n_mma_max; n_warp *= 2)
                            for(int m_warp_num = 1; m_warp_num <= 4; m_warp_num *= 2)
                                for(int n_warp_num = 1; n_warp_num <= 4; n_warp_num *= 2) {

                                    int m_cta = m_warp * m_warp_num;
                                    int n_cta = n_warp * n_warp_num;
                                    int cta_size_in_warp = m_warp_num * n_warp_num;
                                    int cta_size_in_thd  = cta_size_in_warp * WARP_SIZE;

                                    if( m_conv >= 64 && m_cta < 64 ) continue;

                                    int m_cta_num = DivUp(m_conv, m_cta);
                                    int n_cta_num = DivUp(n_conv, n_cta);
                                    int cta_num = m_cta_num * n_cta_num * num_grp;

                                    int kloop_total = flt_hw * DivUp(num_chl_per_grp_pad, k_cta);
                                    int kloop_num = kloop_total;

                                    if( k_cta != GetTileKSize(num_chl_per_grp_pad, kloop_num) ) continue;

                                    if( m_warp == m_mma && n_warp == n_mma ) continue;
                                    if( m_warp == m_mma_max && n_warp == n_mma_max ) continue;
                                    if( m_warp_num == 4 && n_warp_num == 4 ) continue;
                                    if( m_warp_num == 1 && n_warp_num == 1 && k_cta == k_mma_max ) continue;
                                    if(buf_num > kloop_num) continue;

                                    int regs_per_thd = GetSwzlRegsPerThread(type, type_size, m_cta, n_cta, k_cta, m_warp, n_warp, \
                                            m_mma, n_mma, k_mma, k_blk_mma, buf_num, cta_size_in_thd);
                                    int regs_per_cta = regs_per_thd * cta_size_in_thd;
                                    if (regs_per_thd > max_regs_per_thd) continue;
                                    if (regs_per_cta > max_regs_per_cta) continue;

                                    int smem_per_cta = GetSwzlSmemUsage(type, type_size, m_cta, n_cta, k_cta, m_warp, n_warp, \
                                            m_mma, n_mma, buf_num, cta_size_in_warp);
                                    if (smem_per_cta > max_dyn_smem_per_cta) continue;

                                    float eff_score = GetEfficiencyScore(m_cta, n_cta, k_cta, kloop_total, m_conv, n_conv, k_conv);
                                    if(eff_score < 0.5) continue;

                                    float cta_launch_times = 0.f;
                                    float occ_score = GetOccupancyScore(cta_size_in_thd, cta_size_in_warp, \
                                            sm_num, cta_num, regs_per_cta, smem_per_cta,  max_ctas_per_sm, \
                                            max_thds_per_sm, max_regs_per_sm, max_smem_per_sm, cta_launch_times);
                                    if(occ_score < 0.5) continue;

                                    float pip_score = GetSwzlPipelineScore(type_size, cta_launch_times, m_conv, n_conv, k_conv, \
                                            kloop_num, out_w, cta_size_in_thd, cta_size_in_warp, sm_num, m_cta, \
                                            n_cta, k_cta, m_warp, n_warp, buf_num, m_mma, n_mma, k_mma, k_mma_max, \
                                            cpi_mma, cpi_ldg128_l1d, cpi_ldg128_l2, cpi_lds128, cpi_sts32, latency_mma, \
                                            latency_l2_cache, latency_dram);

                                    float score = eff_score + occ_score + pip_score;
                                    nominee.SetSwzlKernelParam(m_cta, n_cta, k_cta, m_warp, n_warp, flt_size, \
                                            buf_num, cta_size_in_thd, smem_per_cta, splitk, splitf, mma_shape);

                                    nominees.push_back(std::make_pair(nominee, score));
                                }

            if(nominees.size() == 0) {
                // nvswzlConv_b128x128_w64x64_k16_buf1
                nominee.SetSwzlKernelParam(128, 128, 16, 64, 64, flt_size, 1, 128, 8192, 1, 1, mma_shape);
                nominees.push_back(std::make_pair(nominee, 0.f));
            }
        }
    }

    std::sort(nominees.begin(), nominees.end(), SortByDescendScore);

    int declare_times        = 0;

    auto mgr = CodeGeneFactorManager::Instance();
    auto gene_factor = mgr->FindKernel(type);

    for(size_t i = 0; i < Min(32, nominees.size()); i++) {
        std::string source = "";
        auto& nominee = nominees[i].first;

        if (nominee.conv_type == "idxn") {
            gene_factor->GeneIdxnKernel(source, nominee.algo_name, nominee.mma_shape, nominee.tiles.flt_size, nominee.tiles.m_cta, nominee.tiles.n_cta, nominee.tiles.m_warp, nominee.tiles.n_warp, nominee.tiles.k_cta, nominee.tiles.k_per_step, declare_times);
            declare_times++;
        } else if (nominee.conv_type == "2spk") {
            gene_factor->Gene2spkKernel(source, nominee.algo_name, nominee.mma_shape, nominee.tiles.flt_size, nominee.tiles.m_cta, nominee.tiles.n_cta, nominee.tiles.m_warp, nominee.tiles.n_warp, nominee.tiles.k_cta, nominee.tiles.k_per_set, nominee.splitk, nominee.splitf, nominee.tiles.buf, declare_times);
            declare_times++;
        } else if (nominee.conv_type == "swzl") {
            gene_factor->GeneSwzlKernel(source, nominee.algo_name, nominee.mma_shape, nominee.tiles.flt_size, nominee.tiles.m_cta, nominee.tiles.n_cta, nominee.tiles.m_warp, nominee.tiles.n_warp, nominee.tiles.k_cta, nominee.splitk, nominee.tiles.buf, declare_times);
            declare_times++;
        }

        sources = sources + source;

        knames.push_back(nominee.algo_name);
        params.push_back(nominee);

    }
#endif

    return ppl::common::RC_SUCCESS;
}

double PPLCUDAConvolutionJitSelectKernel(
    const cudaDeviceProp& device_prop,
    cudaStream_t &stream,
    ppl::common::datatype_t type,
    int4 *d_input,
    int4 *d_flt,
    int4 *d_output,
    int4 *bias,
    int4 *d_temp_buf,
    algo_param_t &algo_param,
    conv_param_t &conv_param,
    fuse_param_t &fuse_param,
    uint64_t workspace)
{
    double elapsed = 0.0f;
#ifdef PPLNN_ENABLE_CUDA_JIT
    std::vector<std::string> knames;
    std::vector<algo_param_t> params;
    std::string sources = "";

    GetFp16ConvKernelNominees(device_prop, type, conv_param, knames, params, sources, false);

    int index = 0;
    std::vector<const char *> compile_params;
    elapsed = AlgoForwardTime(device_prop, stream, knames, sources, index, compile_params, true, type, d_input, d_flt, d_output, bias, d_temp_buf, params, conv_param, fuse_param, workspace);

    algo_param = params[index];
#endif
    return elapsed;
}

float AlgoForwardTime(
    const cudaDeviceProp& device_prop,
    cudaStream_t &stream,
    std::vector<std::string> kname,
    std::string code,
    int &idx,
    std::vector<const char *> compile_params,
    bool include,
    ppl::common::datatype_t type,
    int4 *d_input,
    int4 *d_flt,
    int4 *d_output,
    int4 *bias,
    int4 *d_temp_buf,
    std::vector<algo_param_t> &algo_param,
    conv_param_t &conv_param,
    fuse_param_t &fuse_param,
    uint64_t workspace)
{
    float elapsed = 0;

#ifdef PPLNN_ENABLE_CUDA_JIT
    std::string src_name = kname[0];
    std::string ptx           = CUDANVRTCCompileImpl(std::pair<std::string, std::string>(src_name, code), compile_params, device_prop, include);
    CUmodule module_ptr = nullptr;
    GetKernelFuncImpl(module_ptr, ptx, kname[0]);
    float min_time = FLT_MAX;
    int times      = 1;

    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    for (size_t n = 0; n < kname.size(); n++) {
        CUfunction function = GetKernelFuncImpl(module_ptr, ptx, kname[n]);
        cudaEventRecord(begin, stream);
        for (int i = 0; i < times; i++) {
            PPLCUDAConvolutionForwardJitImp(
                device_prop, stream, function, type, d_input, d_flt, d_output, bias, d_temp_buf, algo_param[n], conv_param, fuse_param);
        }
        cudaEventRecord(end, stream);
        cudaEventSynchronize(begin);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed, begin, end);
        if (elapsed < min_time) {
            min_time = elapsed;
            idx      = n;
        }
    }
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    if (module_ptr) cuModuleUnload(module_ptr);
#endif
    return elapsed;
}


void PPLCUDAConvolutionForwardJitImp(
    const cudaDeviceProp& device_prop,
    cudaStream_t &stream,
    CUfunction function,
    ppl::common::datatype_t type,
    int4 *d_input,
    int4 *d_flt,
    int4 *d_output,
    int4 *bias,
    int4 *d_temp_buf,
    algo_param_t &algo_param,
    conv_param_t &conv_param,
    fuse_param_t &fuse_param)
{
#if __CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 9020
#ifdef PPLNN_ENABLE_CUDA_JIT
    unsigned int splitk = algo_param.splitk;
    unsigned int splitf = algo_param.splitf;

    int pad_size = GetPadSize(type);

    int num_chl_per_grp = conv_param.num_chl / conv_param.num_grp;
    int num_flt_per_grp = conv_param.num_flt / conv_param.num_grp;

    int num_chl_per_grp_pad = Align(num_chl_per_grp, pad_size);
    int num_flt_per_grp_pad = Align(num_flt_per_grp, pad_size);

    int in_hw  = conv_param.in_height * conv_param.in_width;
    int flt_hw = conv_param.flt_height * conv_param.flt_width;
    int out_hw = conv_param.out_height * conv_param.out_width;

    int concat_offset_v8 = fuse_param.concat_offset / pad_size;
    int concat_stride_v8 = fuse_param.concat_stride / pad_size;

    bool is_in_grp_pad  = num_chl_per_grp_pad != num_chl_per_grp && conv_param.num_grp != 1;
    bool is_out_grp_pad = num_flt_per_grp_pad != num_flt_per_grp && conv_param.num_grp != 1;

    uint64_t buf_off_v4 = 0;

    int4 *pad_input  = d_input;
    int4 *pad_output = d_output;

    if (is_in_grp_pad) {
        pad_input = d_temp_buf;
        buf_off_v4 += GetCvtInputSize(type, conv_param, num_chl_per_grp_pad) / (_4INT_TO_INT4_ * _INT_TO_4BYTE_);

        PPLCUDAConvolutionCvtInput(stream, pad_input, d_input, type, conv_param);
    }

    if (is_out_grp_pad) {
        pad_output = d_temp_buf + buf_off_v4;
        buf_off_v4 += getCvtOutputSize(type, conv_param, num_flt_per_grp_pad) / (_4INT_TO_INT4_ * _INT_TO_4BYTE_);
    }

    int4 *final_out = fuse_param.has_concat ? (int4 *)fuse_param.post_concat : pad_output;

    int4 *splitk_buf = d_temp_buf + buf_off_v4;
    int4 *conv_out   = (splitk > 1 || splitf > 1) ? splitk_buf : final_out;

    __half2 clip_min     = __float2half2_rn(fuse_param.clip_min);
    __half2 clip_max     = __float2half2_rn(fuse_param.clip_max);
    __half2 elt_clip_min = __float2half2_rn(fuse_param.elt_clip_min);
    __half2 elt_clip_max = __float2half2_rn(fuse_param.elt_clip_max);
    __half leaky         = __float2half(fuse_param.leaky);
    __half elt_leaky     = __float2half(fuse_param.elt_leaky);

    int tile_n = algo_param.tiles.n_cta;
    int tile_m = algo_param.tiles.m_cta;
    int cta_k  = algo_param.tiles.k_cta;

    dim3 block_size, grid_size;
    block_size.x = algo_param.tiles.cta_size_in_thd;
    block_size.y = 1;
    block_size.z = 1;

    int smem_size = algo_param.tiles.smem_size;

    if(algo_param.conv_type == "swzl") {
        grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, tile_n);
        grid_size.y = DivUp(num_flt_per_grp_pad, tile_m);
    } else {
        grid_size.x = DivUp(conv_param.in_num * conv_param.out_height * conv_param.out_width, tile_m);
        grid_size.y = DivUp(num_flt_per_grp_pad, tile_n);
    }
    grid_size.z = conv_param.num_grp * splitk * splitf * algo_param.gemm_batch;

    const int4 *pre_data  = (const int4 *)fuse_param.pre_data;
    const void *prelu     = (const void *)fuse_param.prelu;
    const void *elt_prelu = (const void *)fuse_param.elt_prelu;

    if(algo_param.conv_type == "idxn") {
        int img_pad_size = pad_size;
        int flt_pad_size = algo_param.tiles.flt_pad_size;

        int out_nhw = out_hw * conv_param.in_num;

        int in_chl_per_grp_pad  = Align(num_chl_per_grp, img_pad_size);
        int flt_chl_per_grp_pad = Align(num_chl_per_grp, flt_pad_size);
        int num_flt_per_grp_pad = Align(num_flt_per_grp, img_pad_size);

        int kloop_num    = DivUp(flt_hw * flt_chl_per_grp_pad, cta_k);
        int koff_num_pad = Align(kloop_num * (cta_k / flt_pad_size), WARP_SIZE);

        void *args[] = {&pad_input, &d_flt, &conv_out, &kloop_num, &koff_num_pad, &in_hw, &out_hw, &flt_hw, &out_nhw, &conv_param.in_height, &conv_param.in_width, &conv_param.in_num, &conv_param.num_grp, &conv_param.num_chl, &num_chl_per_grp, &in_chl_per_grp_pad, &flt_chl_per_grp_pad, &conv_param.flt_height, &conv_param.flt_width, &num_flt_per_grp, &num_flt_per_grp_pad, &conv_param.out_height, &conv_param.out_width, &conv_param.stride_height, &conv_param.stride_width, &conv_param.pad_height, &conv_param.pad_width, &conv_param.hole_height, &conv_param.hole_width, &conv_param.has_bias, &bias, &fuse_param.has_activation, &clip_min, &fuse_param.has_clip, &clip_max, &fuse_param.has_prelu, &prelu, &fuse_param.has_elt, &(pre_data), &fuse_param.has_elt_activation, &elt_clip_min, &fuse_param.has_elt_clip, &elt_clip_max, &fuse_param.has_elt_prelu, &(elt_prelu), &leaky, &elt_leaky, &fuse_param.has_concat, &concat_offset_v8, &concat_stride_v8};

        CUDA_SAFE_CALL(cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, 0, stream, args, 0));
    } else if (algo_param.conv_type == "2spk") {
        int kloop_num = (flt_hw / splitf) * DivUp(num_chl_per_grp_pad, cta_k);

        if(smem_size > MAX_STATIC_SMEM_SIZE_PER_CTA)
            cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size);

        lut_t in_lut, flt_lut;
        int in_lut_size, flt_lut_size;

        InitializeInputLut(in_lut_size, in_lut.idx, conv_param.flt_height, conv_param.flt_width, conv_param.in_height, conv_param.in_width, conv_param.pad_height, conv_param.pad_width, conv_param.hole_height, conv_param.hole_width, num_chl_per_grp_pad, conv_param.num_grp, cta_k, pad_size);

        InitializeFilterLut(flt_lut_size, flt_lut.idx, conv_param.flt_height, conv_param.flt_width, num_chl_per_grp_pad, cta_k, pad_size);
        if (splitk == 1) {
            void *args[] = {&pad_input, &d_flt, &conv_out, &kloop_num, &in_lut, &in_lut_size, &flt_lut, &flt_lut_size, &in_hw, &out_hw, &flt_hw, &splitk, &conv_param.in_height, &conv_param.in_width, &conv_param.in_num, &conv_param.num_grp, &num_chl_per_grp, &num_chl_per_grp_pad, &conv_param.flt_height, &conv_param.flt_width, &num_flt_per_grp, &num_flt_per_grp_pad, &conv_param.out_height, &conv_param.out_width, &conv_param.stride_height, &conv_param.stride_width, &conv_param.pad_height, &conv_param.pad_width, &conv_param.hole_height, &conv_param.hole_width, &conv_param.has_bias, &bias, &fuse_param.has_activation, &clip_min, &fuse_param.has_clip, &clip_max, &fuse_param.has_prelu, &prelu, &fuse_param.has_elt, &(pre_data), &fuse_param.has_elt_activation, &elt_clip_min, &fuse_param.has_elt_clip, &elt_clip_max, &fuse_param.has_elt_prelu, &(elt_prelu), &leaky, &elt_leaky, &fuse_param.has_concat, &concat_offset_v8, &concat_stride_v8};
            CUDA_SAFE_CALL(cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, smem_size, stream, args, 0));
        } else {
            int num_chl_per_spk_head, num_chl_per_spk_tail;

            InitializeNumChlPerSpk(num_chl_per_spk_head, num_chl_per_spk_tail, conv_param.num_chl, conv_param.num_grp, pad_size, cta_k, splitk);

            void *args[] = {&pad_input, &d_flt, &conv_out, &kloop_num, &in_lut, &in_lut_size, &flt_lut, &flt_lut_size, &num_chl_per_spk_head, &num_chl_per_spk_tail, &in_hw, &out_hw, &flt_hw, &splitk, &conv_param.in_height, &conv_param.in_width, &conv_param.in_num, &conv_param.num_grp, &num_chl_per_grp, &num_chl_per_grp_pad, &conv_param.flt_height, &conv_param.flt_width, &num_flt_per_grp, &num_flt_per_grp_pad, &conv_param.out_height, &conv_param.out_width, &conv_param.stride_height, &conv_param.stride_width, &conv_param.pad_height, &conv_param.pad_width, &conv_param.hole_height, &conv_param.hole_width, &conv_param.has_bias, &bias};
            CUDA_SAFE_CALL(cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, smem_size, stream, args, 0));
        }
    } else if (algo_param.conv_type == "swzl") {
        int kloop_num = (flt_hw / splitf) * DivUp(num_chl_per_grp_pad, cta_k);

        if(smem_size > MAX_STATIC_SMEM_SIZE_PER_CTA)
            cuFuncSetAttribute(function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size);

        lut_t in_lut, flt_lut;
        int in_lut_size, flt_lut_size;

        InitializeInputLut(in_lut_size, in_lut.idx, conv_param.flt_height, conv_param.flt_width, conv_param.in_height, conv_param.in_width, conv_param.pad_height, conv_param.pad_width, conv_param.hole_height, conv_param.hole_width, num_chl_per_grp_pad, conv_param.num_grp, cta_k, pad_size);

        InitializeFilterLut(flt_lut_size, flt_lut.idx, conv_param.flt_height, conv_param.flt_width, num_chl_per_grp_pad, cta_k, pad_size);

        void *args[] = {&d_flt, &pad_input, &conv_out, &kloop_num, &in_lut, &in_lut_size, &flt_lut, &flt_lut_size, &in_hw, &out_hw, &flt_hw, &splitk, &conv_param.in_height, &conv_param.in_width, &conv_param.in_num, &conv_param.num_grp, &num_chl_per_grp, &num_chl_per_grp_pad, &conv_param.flt_height, &conv_param.flt_width, &num_flt_per_grp, &num_flt_per_grp_pad, &conv_param.out_height, &conv_param.out_width, &conv_param.stride_height, &conv_param.stride_width, &conv_param.pad_height, &conv_param.pad_width, &conv_param.hole_height, &conv_param.hole_width, &conv_param.has_bias, &bias, &fuse_param.has_activation, &clip_min, &fuse_param.has_clip, &clip_max, &fuse_param.has_prelu, &prelu, &fuse_param.has_elt, &(pre_data), &fuse_param.has_elt_activation, &elt_clip_min, &fuse_param.has_elt_clip, &elt_clip_max, &fuse_param.has_elt_prelu, &(elt_prelu), &leaky, &elt_leaky, &fuse_param.has_concat, &concat_offset_v8, &concat_stride_v8};
        CUDA_SAFE_CALL(cuLaunchKernel(function, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, smem_size, stream, args, 0));
    }

    if (splitk > 1 || splitf > 1) {
        int spk_width_v8  = num_flt_per_grp_pad * conv_param.num_grp / pad_size;
        int spk_height_v1 = out_hw * conv_param.in_num;

        dim3 merge_grid_size, merge_block_size;
        merge_block_size.x = 64;
        merge_block_size.y = 1;
        merge_block_size.z = 1;

        merge_grid_size.x = spk_height_v1;
        merge_grid_size.y = DivUp(spk_width_v8, merge_block_size.x);
        merge_grid_size.z = 1;

        MergeConvSplitResults<<<merge_grid_size, merge_block_size, 0, stream>>>(MERGE_KPARAM_LIST);
    }
    if (is_out_grp_pad) {
        PPLCUDAConvolutionCvtOutput(stream, d_output, final_out, type, conv_param);
    }
#endif
#endif
}
