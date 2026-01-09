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
#include "xtd/math/sqrt.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_sqrt.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::sqrt", "[sqrt][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::sqrt(float)") {
      validate<float, float, xtd::sqrt, mpfr_sqrtf>(device, ulps_single);
    }

    SECTION("double xtd::sqrt(double)") {
      validate<double, double, xtd::sqrt, mpfr_sqrt>(device, ulps_double);
    }

    SECTION("double xtd::sqrt(int)") {
      validate<double, int, xtd::sqrt, mpfr_sqrt>(device, ulps_double);
    }

    SECTION("float xtd::sqrtf(float)") {
      validate<float, float, xtd::sqrtf, mpfr_sqrtf>(device, ulps_single);
    }

    SECTION("float xtd::sqrtf(double)") {
      validate<float, double, xtd::sqrtf, mpfr_sqrtf>(device, ulps_single);
    }

    SECTION("float xtd::sqrtf(int)") {
      validate<float, int, xtd::sqrtf, mpfr_sqrtf>(device, ulps_single);
    }
  }
}
