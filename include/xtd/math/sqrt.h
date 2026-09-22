/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>, Simone Balducci <simone.balducci@cern.ch>, Patrick Kopper <patrick.kopper@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>
#include <cmath>

#include "xtd/internal/concepts.h"
#include "xtd/internal/defines.h"

namespace xtd {
  /* Computes the square root of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t sqrt(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::hsqrt(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::hsqrt(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sqrt(static_cast<sycl::half>(arg));
#elif XTD_HAS_STDFLOAT16
    // standard C/C++ code
#if defined(__clang__) && __has_builtin(__builtin_sqrtf16) && defined(__AVX512FP16__)
    if (std::is_constant_evaluated()) {
      return std::sqrt(static_cast<float>(arg));
    }
    return __builtin_sqrtf16(std::bit_cast<std::float16_t>(arg));
#elif __has_builtin(__builtin_sqrtf)
    if (std::is_constant_evaluated()) {
      return std::sqrt(static_cast<float>(arg));
    }
    return __builtin_sqrtf(static_cast<float>(arg));
#else
    return std::sqrt(static_cast<float>(arg));
#endif
#else
    return std::sqrt(static_cast<float_t>(arg));
#endif
  }

  /* Computes the square root of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float sqrt(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::sqrtf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::sqrtf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sqrt(arg);
#else
    // standard C/C++ code
    return ::sqrtf(arg);
#endif
  }

  /* Computes the square root of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double sqrt(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::sqrt(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::sqrt(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sqrt(arg);
#else
    // standard C/C++ code
    return ::sqrt(arg);
#endif
  }

  /* Computes the square root of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double sqrt(std::integral auto arg) {
    return xtd::sqrt(static_cast<double>(arg));
  }

  /* Computes the square root of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float sqrtf(std::floating_point auto arg) {
    return xtd::sqrt(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float sqrtf(std::integral auto arg) {
    return xtd::sqrt(static_cast<float>(arg));
  }

  /* Computes the square root of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t sqrtf(xtd::floating_point auto arg) {
    return xtd::sqrt(static_cast<float16_t>(arg));
  }

}  // namespace xtd
