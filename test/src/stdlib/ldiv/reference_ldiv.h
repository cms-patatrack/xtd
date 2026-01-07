/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cstdlib>

inline std::ldiv_t reference_ldiv(long n, long d) {
  // If the denominator is 0, return {0, 0} instead of raising a floating point exception.
  return d ? std::ldiv(n, d) : std::ldiv_t{0, 0};
}
