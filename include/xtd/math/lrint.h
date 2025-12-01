/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>, Simone Balducci <simone.balducci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>
#include <cmath>

#include "xtd/internal/defines.h"
#include "xtd/math/fpclassify.h"

namespace xtd {

  /* Computes the nearest integral value to arg as a signed long int, rounding halfway cases to the nearest even integer
   * value.
   * Overflow and plus infinity are clamped to std::numeric_limits<long>::max(), underflow and minus infinity are
   * clamped to std::numeric_limits<long>::min(), while NaN (Not a Number) is converted to 0.
   */
  XTD_DEVICE_FUNCTION inline constexpr long lrint(float arg) {
    switch (xtd::fpclassify(arg)) {
      [[likely]] case FP_NORMAL:
#if defined(XTD_TARGET_CUDA)
        // CUDA device code
        return ::lrintf(arg);
#elif defined(XTD_TARGET_HIP)
        // HIP/ROCm device code
        return ::lrintf(arg);
#elif defined(XTD_TARGET_SYCL)
        // SYCL device code
        return static_cast<long>(sycl::rint(arg));
#else
        // standard C/C++ code
        return ::lrintf(arg);
#endif
      case FP_INFINITE:
        return arg > 0 ? std::numeric_limits<long>::max() : std::numeric_limits<long>::min();
      case FP_ZERO:
        [[fallthrough]];
      case FP_SUBNORMAL:
        [[fallthrough]];
      case FP_NAN:
        [[fallthrough]];
      default:
        return 0;
    }
  }

  /* Computes the nearest integral value to arg as a signed long int, rounding halfway cases to the nearest even integer
   * value.
   * Overflow and plus infinity are clamped to std::numeric_limits<long>::max(), underflow and minus infinity are
   * clamped to std::numeric_limits<long>::min(), while NaN (Not a Number) is converted to 0.
   */
  XTD_DEVICE_FUNCTION inline constexpr long lrint(double arg) {
    switch (xtd::fpclassify(arg)) {
      [[likely]] case FP_NORMAL:
#if defined(XTD_TARGET_CUDA)
        // CUDA device code
        return ::lrint(arg);
#elif defined(XTD_TARGET_HIP)
        // HIP/ROCm device code
        return ::lrint(arg);
#elif defined(XTD_TARGET_SYCL)
        // SYCL device code
        return static_cast<long>(sycl::rint(arg));
#else
        // standard C/C++ code
        return ::lrint(arg);
#endif
      case FP_INFINITE:
        return arg > 0 ? std::numeric_limits<long>::max() : std::numeric_limits<long>::min();
      case FP_ZERO:
        [[fallthrough]];
      case FP_SUBNORMAL:
        [[fallthrough]];
      case FP_NAN:
        [[fallthrough]];
      default:
        return 0;
    }
  }

  /* Computes the nearest integral value to arg as a signed long int, rounding halfway cases to the nearest even integer
   * value.
   * Overflow and plus infinity are clamped to std::numeric_limits<long>::max(), underflow and minus infinity are
   * clamped to std::numeric_limits<long>::min(), while NaN (Not a Number) is converted to 0.
   * Note: integral values are first converted to double precision, so values larger than 2^53 or lower than -2^53 will
   * be approximeted to the closest representalable value.
   */
  XTD_DEVICE_FUNCTION inline constexpr long lrint(std::integral auto arg) {
    return xtd::lrint(static_cast<double>(arg));
  }

  /* Computes the nearest integral value to arg as a signed long int, rounding halfway cases to the nearest even integer
   * value.
   * Overflow and plus infinity are clamped to std::numeric_limits<long>::max(), underflow and minus infinity are
   * clamped to std::numeric_limits<long>::min(), while NaN (Not a Number) is converted to 0.
   * Note: integral values are first converted to single precision, so values larger than 2^24 or lower than -2^24 will
   * be approximeted to the closest representalable value.
   */
  XTD_DEVICE_FUNCTION inline constexpr long lrintf(std::floating_point auto arg) {
    return xtd::lrint(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr long lrintf(std::integral auto arg) {
    return xtd::lrint(static_cast<float>(arg));
  }

}  // namespace xtd
