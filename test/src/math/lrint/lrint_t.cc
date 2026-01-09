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
#include "xtd/math/lrint.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_lrint.h"

TEST_CASE("xtd::lrint", "[lrint][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("long xtd::lrint(float)") {
      validate<long, float, xtd::lrint, mpfr_lrintf>(device);
    }

    SECTION("long xtd::lrint(double)") {
      validate<long, double, xtd::lrint, mpfr_lrint>(device);
    }

    SECTION("long xtd::lrint(int)") {
      validate<long, int, xtd::lrint, mpfr_lrint>(device);
    }

    SECTION("long xtd::lrintf(float)") {
      validate<long, float, xtd::lrintf, mpfr_lrintf>(device);
    }

    SECTION("long xtd::lrintf(double)") {
      validate<long, double, xtd::lrintf, mpfr_lrintf>(device);
    }

    SECTION("long xtd::lrintf(int)") {
      validate<long, int, xtd::lrintf, mpfr_lrintf>(device);
    }
  }
}
