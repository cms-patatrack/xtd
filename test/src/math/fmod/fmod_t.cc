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
#include "xtd/math/fmod.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_fmod.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::fmod", "[fmod][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::fmod(float, float)") {
      validate<float, float, xtd::fmod, mpfr_fmodf>(device, ulps_single);
    }

    SECTION("double xtd::fmod(double, double)") {
      validate<double, double, xtd::fmod, mpfr_fmod>(device, ulps_double);
    }

    SECTION("double xtd::fmod(int, int)") {
      validate<double, int, xtd::fmod, mpfr_fmod>(device, ulps_double);
    }

    SECTION("float xtd::fmodf(float, float)") {
      validate<float, float, xtd::fmodf, mpfr_fmodf>(device, ulps_single);
    }

    SECTION("float xtd::fmodf(double, double)") {
      validate<float, double, xtd::fmodf, mpfr_fmodf>(device, ulps_single);
    }

    SECTION("float xtd::fmodf(int, int)") {
      validate<float, int, xtd::fmodf, mpfr_fmodf>(device, ulps_single);
    }
  }
}
