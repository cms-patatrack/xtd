/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Patrick Kopper <patrick.kopper@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cstdlib>
#include <limits>

inline std::lldiv_t reference_lldiv(long long n, long long d) {
  // If the denominator is 0, return {0, 0} instead of raising a floating point exception.
  if (d == 0) {
    return std::lldiv_t{.quot=0, .rem=0};
  }
  if (n == std::numeric_limits<long long>::min() && d == -1) {
    return std::lldiv_t{.quot=std::numeric_limits<long long>::min(), .rem=0};
  }
  return std::lldiv(n, d);
}
