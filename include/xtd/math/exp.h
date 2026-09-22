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

  /* Computes e raised to the given power, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t exp(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::hexp(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::hexp(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::exp(static_cast<sycl::half>(arg));
#else
    // standard C/C++ code
#if defined(XTD_HAS_STDFLOAT16)
    constexpr bool native = []<typename T>(T) { return requires(T v) { { std::exp(v) } -> std::same_as<T>; }; }(_Float16{});
    using math_t = std::conditional_t<native, _Float16, float>;
#else
    using math_t = float;
#endif
    return std::exp(static_cast<math_t>(arg));
#endif
  }

  /* Computes e raised to the given power, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float exp(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::expf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::expf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::exp(arg);
#else
    // standard C/C++ code
    return ::expf(arg);
#endif
  }

  /* Computes e raised to the given power, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double exp(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::exp(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::exp(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::exp(arg);
#else
    // standard C/C++ code
    return ::exp(arg);
#endif
  }

  /* Computes e raised to the given power, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double exp(std::integral auto arg) {
    return xtd::exp(static_cast<double>(arg));
  }

  /* Computes e raised to the given power, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float expf(std::floating_point auto arg) {
    return xtd::exp(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float expf(std::integral auto arg) {
    return xtd::exp(static_cast<float>(arg));
  }

  /* Computes e raised to the given power, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t expf16(xtd::floating_point auto arg) {
    return xtd::exp(static_cast<float16_t>(arg));
  }

}  // namespace xtd
