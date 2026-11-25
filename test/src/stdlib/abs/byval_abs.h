/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Patrick Kopper <patrick.kopper@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <cmath>

// xtd headers
#include "xtd/internal/float16.h"

namespace byval {
  // Reference implementation of abs() for xtd::float16_t, evaluated in single precision.
  inline xtd::float16_t abs(xtd::float16_t x) { return xtd::float16_t(std::abs(static_cast<float>(x))); }
}  // namespace byval
