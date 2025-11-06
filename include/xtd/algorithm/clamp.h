/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>

#include "xtd/concepts/arithmetic.h"
#include "xtd/internal/defines.h"

namespace xtd {

  /* Clamp x to the range [lo, hi].
   */
  template <xtd::arithmetic T>
  XTD_DEVICE_FUNCTION inline constexpr T clamp(const T x, const T lo, const T hi) {
    return (x < lo) ? lo : (hi < x) ? hi : x;
  }

  /* Clamp x to the range [lo, hi], using the comparison function comp to compare the values.
   */
  template <xtd::arithmetic T, std::predicate<bool, T, T> Compare>
  XTD_DEVICE_FUNCTION inline constexpr T clamp(const T x, const T lo, const T hi, Compare comp) {
    return comp(x, lo) ? lo : comp(hi, x) ? hi : x;
  }

}  // namespace xtd
