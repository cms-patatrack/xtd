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
#include "xtd/math/sin.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_sin.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::sin", "[sin][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::sin(float)") {
      validate<float, float, xtd::sin, mpfr_sinf>(device, ulps_single);
    }

    SECTION("double xtd::sin(double)") {
      validate<double, double, xtd::sin, mpfr_sin>(device, ulps_double);
    }

    SECTION("double xtd::sin(int)") {
      validate<double, int, xtd::sin, mpfr_sin>(device, ulps_double);
    }

    SECTION("float xtd::sinf(float)") {
      validate<float, float, xtd::sinf, mpfr_sinf>(device, ulps_single);
    }

    SECTION("float xtd::sinf(double)") {
      validate<float, double, xtd::sinf, mpfr_sinf>(device, ulps_single);
    }

    SECTION("float xtd::sinf(int)") {
      validate<float, int, xtd::sinf, mpfr_sinf>(device, ulps_single);
    }
  }
}
