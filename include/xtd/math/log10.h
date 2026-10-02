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
  /* Computes the base 10 logarithm of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t log10(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::hlog10(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::hlog10(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::log10(static_cast<sycl::half>(arg));
#elif XTD_HAS_STDFLOAT16
    // standard C/C++ code
#if defined(__clang__) && __has_builtin(__builtin_log10f16) && defined(__AVX512FP16__)
    if (std::is_constant_evaluated()) {
      return std::log10(static_cast<float>(arg));
    }
    return __builtin_log10f16(std::bit_cast<std::float16_t>(arg));
#elif __has_builtin(__builtin_log10f)
    if (std::is_constant_evaluated()) {
      return std::log10(static_cast<float>(arg));
    }
    return __builtin_log10f(static_cast<float>(arg));
#else
    return std::log10(static_cast<float>(arg));
#endif
#else
    return std::log10(static_cast<float_t>(arg));
#endif
  }

  /* Computes the base 10 logarithm of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float log10(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::log10f(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::log10f(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::log10(arg);
#else
    // standard C/C++ code
    return ::log10f(arg);
#endif
  }

  /* Computes the base 10 logarithm of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double log10(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::log10(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::log10(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::log10(arg);
#else
    // standard C/C++ code
    return ::log10(arg);
#endif
  }

  /* Computes the base 10 logarithm of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double log10(std::integral auto arg) {
    return xtd::log10(static_cast<double>(arg));
  }

  /* Computes the base 10 logarithm of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float log10f(std::floating_point auto arg) {
    return xtd::log10(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float log10f(std::integral auto arg) {
    return xtd::log10(static_cast<float>(arg));
  }

  /* Computes the base 10 logarithm of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t log10f(xtd::floating_point auto arg) {
    return xtd::log10(static_cast<float16_t>(arg));
  }

}  // namespace xtd
