/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cstdlib>

inline std::div_t reference_div(int n, int d) {
  // If the denominator is 0, return {0, 0} instead of raising a floating point exception.
  return d ? std::div(n, d) : std::div_t{0, 0};
}
