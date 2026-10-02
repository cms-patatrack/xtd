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

  /* Computes the hyperbolic sine of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t sinh(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return __float2half(::sinhf(__half2float(arg)));
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    // AMD uses a suboptimal range reduction, use full float evaluation
    return __float2half(::sinhf(__half2float(arg)));
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sinh(static_cast<float>(arg));
#elif XTD_HAS_STDFLOAT16
    // standard C/C++ code
#if defined(__clang__) && __has_builtin(__builtin_sinhf16) && defined(__AVX512FP16__)
    if (std::is_constant_evaluated()) {
      return std::sinh(static_cast<float>(arg));
    }
    return __builtin_sinhf16(std::bit_cast<std::float16_t>(arg));
#elif __has_builtin(__builtin_sinhf)
    if (std::is_constant_evaluated()) {
      return std::sinh(static_cast<float>(arg));
    }
    return __builtin_sinhf(static_cast<float>(arg));
#else
    return std::sinh(static_cast<float>(arg));
#endif
#else
    return float16_t(std::sinh(static_cast<float_t>(arg)));
#endif
  }

  /* Computes the hyperbolic sine of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float sinh(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::sinhf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::sinhf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sinh(arg);
#else
    // standard C/C++ code
    return ::sinhf(arg);
#endif
  }

  /* Computes the hyperbolic sine of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double sinh(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::sinh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::sinh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sinh(arg);
#else
    // standard C/C++ code
    return ::sinh(arg);
#endif
  }

  /* Computes the hyperbolic sine of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double sinh(std::integral auto arg) {
    return xtd::sinh(static_cast<double>(arg));
  }

  /* Computes the hyperbolic sine of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float sinhf(std::floating_point auto arg) {
    return xtd::sinh(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float sinhf(std::integral auto arg) {
    return xtd::sinh(static_cast<float>(arg));
  }

}  // namespace xtd
