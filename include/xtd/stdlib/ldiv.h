/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>, Simone Balducci <simone.balducci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdlib>

#include "xtd/internal/defines.h"

namespace xtd {

  /* Computes the quotient and reminder of the integer division `numerator / denominator`,
   * and returns them in the `quot` and `rem` members of an `ldiv_t` struct.
   * If the denominator is 0, does not raise a floating point exception and returns { 0, 0 }.
   */
  XTD_DEVICE_FUNCTION inline constexpr ::ldiv_t ldiv(long numerator, long denominator) {
    ::ldiv_t result;
    if (denominator != 0) {
      result.quot = numerator / denominator;
      result.rem = numerator % denominator;
    } else {
      result.quot = 0;
      result.rem = 0;
    }
    return result;
  }

}  // namespace xtd
