/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cstdlib>

inline std::lldiv_t reference_lldiv(long long n, long long d) {
  // If the denominator is 0, return {0, 0} instead of raising a floating point exception.
  return d ? std::lldiv(n, d) : std::lldiv_t{0, 0};
}
