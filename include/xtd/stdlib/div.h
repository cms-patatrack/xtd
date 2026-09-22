/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>, Simone Balducci <simone.balducci@cern.ch>,
 *          Patrick Kopper <patrick.kopper@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdlib>

#include "xtd/internal/defines.h"

namespace xtd {

  /* Computes the quotient and reminder of the integer division `numerator / denominator`,
   * and returns them in the `quot` and `rem` members of an `div_t` struct.
   * If the denominator is 0, does not raise a floating point exception and returns { 0, 0 }.
   */
  XTD_DEVICE_FUNCTION inline constexpr ::div_t div(int numerator, int denominator) {
    ::div_t result{};
    if (denominator == 0) {
      return result;
    }
    if (numerator == std::numeric_limits<int>::min() && denominator == -1) {
      result.quot = std::numeric_limits<int>::min();
      return result;
    }
    result.quot = numerator / denominator;
    result.rem = numerator % denominator;
    return result;
  }

}  // namespace xtd
