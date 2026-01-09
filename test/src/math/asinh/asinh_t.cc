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
#include "xtd/math/asinh.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_asinh.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::asinh", "[asinh][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::asinh(float)") {
      validate<float, float, xtd::asinh, mpfr_asinhf>(device, ulps_single);
    }

    SECTION("double xtd::asinh(double)") {
      validate<double, double, xtd::asinh, mpfr_asinh>(device, ulps_double);
    }

    SECTION("double xtd::asinh(int)") {
      validate<double, int, xtd::asinh, mpfr_asinh>(device, ulps_double);
    }

    SECTION("float xtd::asinhf(float)") {
      validate<float, float, xtd::asinhf, mpfr_asinhf>(device, ulps_single);
    }

    SECTION("float xtd::asinhf(double)") {
      validate<float, double, xtd::asinhf, mpfr_asinhf>(device, ulps_single);
    }

    SECTION("float xtd::asinhf(int)") {
      validate<float, int, xtd::asinhf, mpfr_asinhf>(device, ulps_single);
    }
  }
}
