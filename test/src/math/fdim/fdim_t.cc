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
#include "xtd/math/fdim.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_fdim.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::fdim", "[fdim][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::fdim(float, float)") {
      validate<float, float, xtd::fdim, mpfr_fdimf>(device, ulps_single);
    }

    SECTION("double xtd::fdim(double, double)") {
      validate<double, double, xtd::fdim, mpfr_fdim>(device, ulps_double);
    }

    SECTION("double xtd::fdim(int, int)") {
      validate<double, int, xtd::fdim, mpfr_fdim>(device, ulps_double);
    }

    SECTION("float xtd::fdimf(float, float)") {
      validate<float, float, xtd::fdimf, mpfr_fdimf>(device, ulps_single);
    }

    SECTION("float xtd::fdimf(double, double)") {
      validate<float, double, xtd::fdimf, mpfr_fdimf>(device, ulps_single);
    }

    SECTION("float xtd::fdimf(int, int)") {
      validate<float, int, xtd::fdimf, mpfr_fdimf>(device, ulps_single);
    }
  }
}
