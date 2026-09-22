/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>, Patrick Kopper <patrick.kopper@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#if __cplusplus < 202002L
#error "xtd requires C++20 or later. Please compile with -std=c++20."
#endif

// XTD_DEVICE_FUNCTION
#if defined(__CUDACC__) || defined(__HIPCC__)
// CUDA or HIP/ROCm compiler
#define XTD_DEVICE_FUNCTION __host__ __device__
#else
// SYCL or standard C++ code
#define XTD_DEVICE_FUNCTION
#endif

// XTD_TARGET_...
#if defined(__CUDA_ARCH__)
// CUDA device code
#define XTD_TARGET_CUDA
#elif defined(__HIP_DEVICE_COMPILE__)
// HIP/ROCm device code
#define XTD_TARGET_HIP
#elif defined(__SYCL_DEVICE_ONLY__)
// SYCL device code
#define XTD_TARGET_SYCL
#else
// standard C++ code
#define XTD_TARGET_CPU
#endif

// backend runtime include files
#if defined(XTD_TARGET_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(XTD_TARGET_HIP)
#include <hip/hip_runtime.h>
#endif

#if defined(XTD_TARGET_SYCL)
#include <sycl/sycl.hpp>
#endif

#if defined(__STDCPP_FLOAT16_T__) && __has_include(<stdfloat>)
    #include <stdfloat>
    #define XTD_HAS_STDFLOAT16 1
#elif defined(__clang__) && defined(__FLT16_MAX__)
    namespace std { using float16_t = _Float16; }
    #define XTD_HAS_STDFLOAT16 1
#else
    #define XTD_HAS_STDFLOAT16 0
#endif
