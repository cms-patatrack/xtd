/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#include <catch.hpp>

// xtd headers
#include "xtd/math/fpclassify.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "reference_fpclassify.h"

TEST_CASE("xtd::fpclassify", "[fpclassify][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("int xtd::fpclassify(float)") {
      validate<int, float, xtd::fpclassify, reference_fpclassify>(device);
    }

    SECTION("int xtd::fpclassify(double)") {
      validate<int, double, xtd::fpclassify, reference_fpclassify>(device);
    }

    SECTION("int xtd::fpclassify(int)") {
      validate<int, int, xtd::fpclassify, reference_fpclassify>(device);
    }
  }
}
