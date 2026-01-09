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
#include "xtd/math/remquo.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate_remquo.h"
#include "mpfr_remquo.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::remquo", "[remquo][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::remquo(float, float, int*)") {
      validate_remquo<float, float, xtd::remquo, mpfr_remquof>(device, ulps_single);
    }

    SECTION("double xtd::remquo(double, double, int*)") {
      validate_remquo<double, double, xtd::remquo, mpfr_remquo>(device, ulps_double);
    }

    SECTION("double xtd::remquo(int, int, int*)") {
      validate_remquo<double, int, xtd::remquo, mpfr_remquo>(device, ulps_double);
    }

    SECTION("float xtd::remquof(float, float, int*)") {
      validate_remquo<float, float, xtd::remquof, mpfr_remquof>(device, ulps_single);
    }

    SECTION("float xtd::remquof(double, double, int*)") {
      validate_remquo<float, double, xtd::remquof, mpfr_remquof>(device, ulps_single);
    }

    SECTION("float xtd::remquof(int, int, int*)") {
      validate_remquo<float, int, xtd::remquof, mpfr_remquof>(device, ulps_single);
    }
  }
}
