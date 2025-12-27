/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#include <catch.hpp>

// xtd headers
#include "xtd/math/isfinite.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "reference_isfinite.h"

TEST_CASE("xtd::isfinite", "[isfinite][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("int xtd::isfinite(float)") {
      validate<int, float, xtd::isfinite, reference_isfinite>(device);
    }

    SECTION("int xtd::isfinite(double)") {
      validate<int, double, xtd::isfinite, reference_isfinite>(device);
    }

    SECTION("int xtd::isfinite(int)") {
      validate<int, int, xtd::isfinite, reference_isfinite>(device);
    }
  }
}
