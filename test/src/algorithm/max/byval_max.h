/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <algorithm>

// xtd headers
#include "xtd/concepts/arithmetic.h"

namespace byval {
  template <xtd::arithmetic T>
  T max(T a, T b) {
    return std::max(a, b);
  }

  inline xtd::float16_t max(xtd::float16_t a, xtd::float16_t b) {
    return xtd::float16_t(std::max(static_cast<float>(a), static_cast<float>(b)));
  }
}  // namespace byval
