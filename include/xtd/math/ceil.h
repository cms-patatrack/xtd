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

  /* Computes the smallest integral value that is not less than arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t ceil(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::hceil(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::hceil(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::ceil(static_cast<sycl::half>(arg));
#else
    // standard C/C++ code
#if defined(XTD_HAS_STDFLOAT16)
    constexpr bool native = []<typename T>(T) { return requires(T v) { { std::ceil(v) } -> std::same_as<T>; }; }(_Float16{});
    using math_t = std::conditional_t<native, _Float16, float>;
#else
    using math_t = float;
#endif
    return std::ceil(static_cast<math_t>(arg));
#endif
  }

  /* Computes the smallest integral value that is not less than arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float ceil(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::ceilf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::ceilf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::ceil(arg);
#else
    // standard C/C++ code
    return ::ceilf(arg);
#endif
  }

  /* Computes the smallest integral value that is not less than arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double ceil(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::ceil(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::ceil(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::ceil(arg);
#else
    // standard C/C++ code
    return ::ceil(arg);
#endif
  }

  /* Computes the smallest integral value that is not less than arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double ceil(std::integral auto arg) {
    return xtd::ceil(static_cast<double>(arg));
  }

  /* Computes the smallest integral value that is not less than arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float ceilf(std::floating_point auto arg) {
    return xtd::ceil(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float ceilf(std::integral auto arg) {
    return xtd::ceil(static_cast<float>(arg));
  }

  /* Computes the smallest integral value that is not less than arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t ceilf16(xtd::floating_point auto arg) {
    return xtd::ceil(static_cast<float16_t>(arg));
  }

}  // namespace xtd
