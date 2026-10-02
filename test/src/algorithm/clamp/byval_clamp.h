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
  T clamp(T x, T lo, T hi) {
    return std::clamp(x, lo, hi);
  }

  inline xtd::float16_t clamp(xtd::float16_t x, xtd::float16_t lo, xtd::float16_t hi) {
    return xtd::float16_t(std::clamp(static_cast<float>(x), static_cast<float>(lo), static_cast<float>(hi)));
  }
}  // namespace byval
