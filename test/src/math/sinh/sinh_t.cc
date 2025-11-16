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
#include "xtd/math/sinh.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_sinh.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::sinh", "[sinh][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::sinh(float)") {
      validate<float, float, xtd::sinh, mpfr_sinhf>(device, ulps_single);
    }

    SECTION("double xtd::sinh(double)") {
      validate<double, double, xtd::sinh, mpfr_sinh>(device, ulps_double);
    }

    SECTION("double xtd::sinh(int)") {
      validate<double, int, xtd::sinh, mpfr_sinh>(device, ulps_double);
    }

    SECTION("float xtd::sinhf(float)") {
      validate<float, float, xtd::sinhf, mpfr_sinhf>(device, ulps_single);
    }

    SECTION("float xtd::sinhf(double)") {
      validate<float, double, xtd::sinhf, mpfr_sinhf>(device, ulps_single);
    }

    SECTION("float xtd::sinhf(int)") {
      validate<float, int, xtd::sinhf, mpfr_sinhf>(device, ulps_single);
    }
  }
}
