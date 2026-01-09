/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#include <catch.hpp>

// xtd headers
#include "xtd/math/isnormal.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "reference_isnormal.h"

TEST_CASE("xtd::isnormal", "[isnormal][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("int xtd::isnormal(float)") {
      validate<int, float, xtd::isnormal, reference_isnormal>(device);
    }

    SECTION("int xtd::isnormal(double)") {
      validate<int, double, xtd::isnormal, reference_isnormal>(device);
    }

    SECTION("int xtd::isnormal(int)") {
      validate<int, int, xtd::isnormal, reference_isnormal>(device);
    }
  }
}
