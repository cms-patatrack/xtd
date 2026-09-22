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

  /* Computes the inverse hyperbolic sine of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t asinh(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return __float2half(::asinhf(__half2float(arg)));
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return __float2half(::asinhf(__half2float(arg)));
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::asinh(static_cast<sycl::half>(arg));
#elif XTD_HAS_STDFLOAT16
    // standard C/C++ code
#if defined(__clang__) && __has_builtin(__builtin_asinhf16) && defined(__AVX512FP16__)
    if (std::is_constant_evaluated()) {
      return std::asinh(static_cast<float>(arg));
    }
    return __builtin_asinhf16(std::bit_cast<std::float16_t>(arg));
#elif __has_builtin(__builtin_asinhf)
    if (std::is_constant_evaluated()) {
      return std::asinh(static_cast<float>(arg));
    }
    return __builtin_asinhf(static_cast<float>(arg));
#else
    return std::asinh(static_cast<float>(arg));
#endif
#else
    return std::asinh(static_cast<float_t>(arg));
#endif
  }

  /* Computes the inverse hyperbolic sine of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float asinh(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::asinhf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::asinhf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::asinh(arg);
#else
    // standard C/C++ code
    return ::asinhf(arg);
#endif
  }

  /* Computes the inverse hyperbolic sine of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double asinh(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::asinh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::asinh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::asinh(arg);
#else
    // standard C/C++ code
    return ::asinh(arg);
#endif
  }

  /* Computes the inverse hyperbolic sine of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double asinh(std::integral auto arg) {
    return xtd::asinh(static_cast<double>(arg));
  }

  /* Computes the inverse hyperbolic sine of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float asinhf(std::floating_point auto arg) {
    return xtd::asinh(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float asinhf(std::integral auto arg) {
    return xtd::asinh(static_cast<float>(arg));
  }

  /* Computes the inverse hyperbolic sine of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t asinhf(xtd::floating_point auto arg) {
    return xtd::asinh(static_cast<float16_t>(arg));
  }

}  // namespace xtd
