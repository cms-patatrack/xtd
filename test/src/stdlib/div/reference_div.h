/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Patrick Kopper <patrick.kopper@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cstdlib>
#include <limits>

inline std::div_t reference_div(int n, int d) {
  // If the denominator is 0, return {0, 0} instead of raising a floating point exception.
  if (d == 0) {
    return std::div_t{.quot=0, .rem=0};
  }
  if (n == std::numeric_limits<int>::min() && d == -1) {
    return std::div_t{.quot=std::numeric_limits<int>::min(), .rem=0};
  }
  return std::div(n, d);
}
