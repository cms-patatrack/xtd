/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>, Patrick Kopper <patrick.kopper@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <cmath>

// xtd headers
#include "xtd/internal/float16.h"

namespace byval {

  inline xtd::float16_t abs(xtd::float16_t x) {
    return std::abs(static_cast<float>(x));
  }

}  // namespace byval
