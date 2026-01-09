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
#include "xtd/math/lround.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_lround.h"

TEST_CASE("xtd::lround", "[lround][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("long xtd::lround(float)") {
      validate<long, float, xtd::lround, mpfr_lroundf>(device);
    }

    SECTION("long xtd::lround(double)") {
      validate<long, double, xtd::lround, mpfr_lround>(device);
    }

    SECTION("long xtd::lround(int)") {
      validate<long, int, xtd::lround, mpfr_lround>(device);
    }

    SECTION("long xtd::lroundf(float)") {
      validate<long, float, xtd::lroundf, mpfr_lroundf>(device);
    }

    SECTION("long xtd::lroundf(double)") {
      validate<long, double, xtd::lroundf, mpfr_lroundf>(device);
    }

    SECTION("long xtd::lroundf(int)") {
      validate<long, int, xtd::lroundf, mpfr_lroundf>(device);
    }
  }
}
