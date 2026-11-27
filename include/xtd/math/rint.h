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
  /* Computes the nearest integral value to arg in half precision, rounding halfway cases to the nearest even integer value.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t rint(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::hrint(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::hrint(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::rint(static_cast<sycl::half>(arg));
#else
    // standard C/C++ code
#if defined(XTD_HAS_STDFLOAT16)
    constexpr bool native = []<typename T>(T) { return requires(T v) { { std::rint(v) } -> std::same_as<T>; }; }(_Float16{});
    using math_t = std::conditional_t<native, _Float16, float>;
#else
    using math_t = float;
#endif
    return std::rint(static_cast<math_t>(arg));
#endif
  }

  /* Computes the nearest integral value to arg in single precision, rounding halfway cases to the nearest even integer value.
   */
  XTD_DEVICE_FUNCTION inline constexpr float rint(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::rintf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::rintf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::rint(arg);
#else
    // standard C/C++ code
    return ::rintf(arg);
#endif
  }

  /* Computes the nearest integral value to arg in double precision, rounding halfway cases to the nearest even integer value.
   */
  XTD_DEVICE_FUNCTION inline constexpr double rint(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::rint(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::rint(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::rint(arg);
#else
    // standard C/C++ code
    return ::rint(arg);
#endif
  }

  /* Computes the nearest integral value to arg in double precision, rounding halfway cases to the nearest even integer value.
   */
  XTD_DEVICE_FUNCTION inline constexpr double rint(std::integral auto arg) {
    return xtd::rint(static_cast<double>(arg));
  }

  /* Computes the nearest integral value to arg in single precision, rounding halfway cases to the nearest even integer value.
   */
  XTD_DEVICE_FUNCTION inline constexpr float rintf(std::floating_point auto arg) {
    return xtd::rint(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float rintf(std::integral auto arg) {
    return xtd::rint(static_cast<float>(arg));
  }

  /* Computes the nearest integral value to arg in half precision, rounding halfway cases to the nearest even integer value.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t rintf16(xtd::floating_point auto arg) {
    return xtd::rint(static_cast<float16_t>(arg));
  }

}  // namespace xtd
