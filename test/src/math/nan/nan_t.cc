/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cmath>
#include <vector>

// Catch2 headers
#include <catch.hpp>

// xtd headers
#include "xtd/math/nan.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "reference_nan.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::nan", "[nan][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::nanf(uint32_t)") {
      validate<float, uint32_t, xtd::nanf, reference_nanf>(device);
    }

    SECTION("double xtd::nan(uint64_t)") {
      validate<double, uint64_t, xtd::nan, reference_nan>(device);
    }
  }
}
