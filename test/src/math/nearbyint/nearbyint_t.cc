/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cmath>
#include <vector>

// Catch2 headers
#include <catch.hpp>

// xtd headers
#include "xtd/math/nearbyint.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_nearbyint.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::nearbyint", "[nearbyint][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::nearbyint(float)") {
      validate<float, float, xtd::nearbyint, mpfr_nearbyintf>(device, ulps_single);
    }

    SECTION("double xtd::nearbyint(double)") {
      validate<double, double, xtd::nearbyint, mpfr_nearbyint>(device, ulps_double);
    }

    SECTION("double xtd::nearbyint(int)") {
      validate<double, int, xtd::nearbyint, mpfr_nearbyint>(device, ulps_double);
    }

    SECTION("float xtd::nearbyintf(float)") {
      validate<float, float, xtd::nearbyintf, mpfr_nearbyintf>(device, ulps_single);
    }

    SECTION("float xtd::nearbyintf(double)") {
      validate<float, double, xtd::nearbyintf, mpfr_nearbyintf>(device, ulps_single);
    }

    SECTION("float xtd::nearbyintf(int)") {
      validate<float, int, xtd::nearbyintf, mpfr_nearbyintf>(device, ulps_single);
    }
  }
}
