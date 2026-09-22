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
  /* The concept arithmetic<T> is satisfied if and only if T is an integral type or a floating point type.
   */
  template <class T>
  concept arithmetic = std::integral<T> or std::floating_point<T>;

  template <arithmetic T>
  T abs(T x, T lo, T hi) {
    return std::abs(x);
  }

  inline xtd::float16_t abs(xtd::float16_t x) {
    return xtd::float16_t(std::abs(static_cast<float>(x)));
  }
}  // namespace byval

