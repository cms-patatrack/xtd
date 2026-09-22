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

  /* Computes the cube root of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t cbrt(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return __float2half(::cbrtf(__half2float(arg)));
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return __float2half(::cbrtf(__half2float(arg)));
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cbrt(static_cast<sycl::half>(arg));
#elif XTD_HAS_STDFLOAT16
    // standard C/C++ code
#if defined(__clang__) && __has_builtin(__builtin_cbrtf16) && defined(__AVX512FP16__)
    if (std::is_constant_evaluated()) {
      return std::cbrt(static_cast<float>(arg));
    }
    return __builtin_cbrtf16(std::bit_cast<std::float16_t>(arg));
#elif __has_builtin(__builtin_cbrtf)
    if (std::is_constant_evaluated()) {
      return std::cbrt(static_cast<float>(arg));
    }
    return __builtin_cbrtf(static_cast<float>(arg));
#else
    return std::cbrt(static_cast<float>(arg));
#endif
#else
    return std::cbrt(static_cast<float_t>(arg));
#endif
  }

  /* Computes the cube root of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float cbrt(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::cbrtf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::cbrtf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cbrt(arg);
#else
    // standard C/C++ code
    return ::cbrtf(arg);
#endif
  }

  /* Computes the cube root of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double cbrt(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::cbrt(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::cbrt(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cbrt(arg);
#else
    // standard C/C++ code
    return ::cbrt(arg);
#endif
  }

  /* Computes the cube root of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double cbrt(std::integral auto arg) {
    return xtd::cbrt(static_cast<double>(arg));
  }

  /* Computes the cube root of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float cbrtf(std::floating_point auto arg) {
    return xtd::cbrt(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float cbrtf(std::integral auto arg) {
    return xtd::cbrt(static_cast<float>(arg));
  }

  /* Computes the cube root of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t cbrtf(xtd::floating_point auto arg) {
    return xtd::cbrt(static_cast<float16_t>(arg));
  }

}  // namespace xtd
