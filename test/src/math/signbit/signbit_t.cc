/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#include <catch.hpp>

// xtd headers
#include "xtd/math/signbit.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "reference_signbit.h"

TEST_CASE("xtd::signbit", "[signbit][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("int xtd::signbit(float)") {
      validate<int, float, xtd::signbit, reference_signbit>(device);
    }

    SECTION("int xtd::signbit(double)") {
      validate<int, double, xtd::signbit, reference_signbit>(device);
    }

    SECTION("int xtd::signbit(int)") {
      validate<int, int, xtd::signbit, reference_signbit>(device);
    }

    SECTION("int xtd::signbit(unsigned int)") {
      validate<int, unsigned int, xtd::signbit, reference_signbit>(device);
    }
  }
}
