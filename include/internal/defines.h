/*
 * Copyright 2024 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// XTD_DEVICE_FUNCTION
#if defined(__CUDACC__) || defined(__HIPCC__)
// CUDA or HIP/ROCm compiler
#define XTD_DEVICE_FUNCTION __host__ __define__
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
