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

  /* Computes 2 raised to the given power, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t exp2(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::hexp2(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::hexp2(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::exp2(static_cast<sycl::half>(arg));
#else
    // standard C/C++ code
#if defined(XTD_HAS_STDFLOAT16)
    constexpr bool native = []<typename T>(T) { return requires(T v) { { std::exp2(v) } -> std::same_as<T>; }; }(_Float16{});
    using math_t = std::conditional_t<native, _Float16, float>;
#else
    using math_t = float;
#endif
    return std::exp2(static_cast<math_t>(arg));
#endif
  }

  /* Computes 2 raised to the given power, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float exp2(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::exp2f(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::exp2f(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::exp2(arg);
#else
    // standard C/C++ code
    return ::exp2f(arg);
#endif
  }

  /* Computes 2 raised to the given power, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double exp2(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::exp2(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::exp2(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::exp2(arg);
#else
    // standard C/C++ code
    return ::exp2(arg);
#endif
  }

  /* Computes 2 raised to the given power, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double exp2(std::integral auto arg) {
    return xtd::exp2(static_cast<double>(arg));
  }

  /* Computes 2 raised to the given power, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float exp2f(std::floating_point auto arg) {
    return xtd::exp2(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float exp2f(std::integral auto arg) {
    return xtd::exp2(static_cast<float>(arg));
  }

  /* Computes 2 raised to the given power, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t exp2f16(xtd::floating_point auto arg) {
    return xtd::exp2(static_cast<float16_t>(arg));
  }

}  // namespace xtd
