/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <concepts>

// XTD concepts
#include "xtd/internal/concepts.h"
#include "xtd/internal/float16.h"

namespace xtd {

  /* The concept arithmetic<T> is satisfied if and only if T is an integral type or a floating point type.
   */
  template <class T>
  concept arithmetic = std::integral<T> or xtd::floating_point<T>;

}  // namespace xtd
