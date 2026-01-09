/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#include <catch.hpp>

// xtd headers
#include "xtd/math/isnan.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "reference_isnan.h"

TEST_CASE("xtd::isnan", "[isnan][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("int xtd::isnan(float)") {
      validate<int, float, xtd::isnan, reference_isnan>(device);
    }

    SECTION("int xtd::isnan(double)") {
      validate<int, double, xtd::isnan, reference_isnan>(device);
    }

    SECTION("int xtd::isnan(int)") {
      validate<int, int, xtd::isnan, reference_isnan>(device);
    }
  }
}
