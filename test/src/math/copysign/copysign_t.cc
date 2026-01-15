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
#include "xtd/math/copysign.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "reference_copysign.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::copysign", "[copysign][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::copysign(float, float)") {
      validate<float, float, xtd::copysign, reference_copysignf>(device, ulps_single);
    }

    SECTION("double xtd::copysign(double, double)") {
      validate<double, double, xtd::copysign, reference_copysign>(device, ulps_double);
    }

    SECTION("double xtd::copysign(int, int)") {
      validate<double, int, xtd::copysign, reference_copysign>(device, ulps_double);
    }

    SECTION("float xtd::copysignf(float, float)") {
      validate<float, float, xtd::copysignf, reference_copysignf>(device, ulps_single);
    }

    SECTION("float xtd::copysignf(double, double)") {
      validate<float, double, xtd::copysignf, reference_copysignf>(device, ulps_single);
    }

    SECTION("float xtd::copysignf(int, int)") {
      validate<float, int, xtd::copysignf, reference_copysignf>(device, ulps_single);
    }
  }
}
