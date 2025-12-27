/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <concepts>

inline constexpr int reference_isinf(std::floating_point auto arg) {
  return __builtin_isinf(arg) ? arg > 0 ? 1 : -1 : 0;
}

inline constexpr int reference_isinf(std::integral auto arg) {
  return 0;
}
