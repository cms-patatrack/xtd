/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#include <catch.hpp>

// xtd headers
#include "xtd/math/isinf.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "reference_isinf.h"

TEST_CASE("xtd::isinf", "[isinf][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("int xtd::isinf(float)") {
      validate<int, float, xtd::isinf, reference_isinf>(device);
    }

    SECTION("int xtd::isinf(double)") {
      validate<int, double, xtd::isinf, reference_isinf>(device);
    }

    SECTION("int xtd::isinf(int)") {
      validate<int, int, xtd::isinf, reference_isinf>(device);
    }
  }
}
