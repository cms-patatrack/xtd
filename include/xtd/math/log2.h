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
  /* Computes the base 2 logarithm of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t log2(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::hlog2(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::hlog2(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::log2(static_cast<sycl::half>(arg));
#elif XTD_HAS_STDFLOAT16
    // standard C/C++ code
#if defined(__clang__) && __has_builtin(__builtin_log2f16) && defined(__AVX512FP16__)
    if (std::is_constant_evaluated()) {
      return std::log2(static_cast<float>(arg));
    }
    return __builtin_log2f16(std::bit_cast<std::float16_t>(arg));
#elif __has_builtin(__builtin_log2f)
    if (std::is_constant_evaluated()) {
      return std::log2(static_cast<float>(arg));
    }
    return __builtin_log2f(static_cast<float>(arg));
#else
    return std::log2(static_cast<float>(arg));
#endif
#else
    return std::log2(static_cast<float_t>(arg));
#endif
  }

  /* Computes the base 2 logarithm of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float log2(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::log2f(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::log2f(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::log2(arg);
#else
    // standard C/C++ code
    return ::log2f(arg);
#endif
  }

  /* Computes the base 2 logarithm of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double log2(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::log2(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::log2(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::log2(arg);
#else
    // standard C/C++ code
    return ::log2(arg);
#endif
  }

  /* Computes the base 2 logarithm of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double log2(std::integral auto arg) {
    return xtd::log2(static_cast<double>(arg));
  }

  /* Computes the base 2 logarithm of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float log2f(std::floating_point auto arg) {
    return xtd::log2(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float log2f(std::integral auto arg) {
    return xtd::log2(static_cast<float>(arg));
  }

  /* Computes the base 2 logarithm of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t log2f(xtd::floating_point auto arg) {
    return xtd::log2(static_cast<float16_t>(arg));
  }

}  // namespace xtd
