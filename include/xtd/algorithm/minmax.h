/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>
#include <utility>

#include "xtd/concepts/arithmetic.h"
#include "xtd/internal/defines.h"

namespace xtd {

  /* Returns the smaller and greater of the given values, using operator< to
   * compare the values.
   */
  template <xtd::arithmetic T>
  XTD_DEVICE_FUNCTION inline constexpr std::pair<T, T> minmax(const T x, const T y) {
    return (y < x) ? std::pair(y, x) : std::pair(x, y);
  }

  /* Returns the smaller and greater of the given values, using the comparison
   * function comp to compare the values.
   */
  template <xtd::arithmetic T, std::predicate<bool, T, T> Compare>
  XTD_DEVICE_FUNCTION inline constexpr std::pair<T, T> minmax(const T x, const T y, Compare comp) {
    return (comp(y, x)) ? std::pair(y, x) : std::pair(x, y);
  }

}  // namespace xtd
