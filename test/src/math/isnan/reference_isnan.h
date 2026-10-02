/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <concepts>
#include "xtd/internal/defines.h"

inline constexpr int reference_isnan(xtd::float16_t arg) {
  return __builtin_isnan(static_cast<float>(arg)) ? 1 : 0;
}

inline constexpr int reference_isnan(std::floating_point auto arg) {
  return __builtin_isnan(arg) ? 1 : 0;
}

inline constexpr int reference_isnan([[maybe_unused]] std::integral auto arg) {
  return 0;
}
