/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cmath>
#include <concepts>

inline constexpr int reference_fpclassify(std::floating_point auto arg) {
  return __builtin_fpclassify(FP_NAN, FP_INFINITE, FP_NORMAL, FP_SUBNORMAL, FP_ZERO, arg);
}

inline constexpr int reference_fpclassify(std::integral auto arg) {
  return arg ? FP_NORMAL : FP_ZERO;
}
