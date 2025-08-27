/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cmath>
#include <vector>

// Catch2 headers
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include "xtd/math/fmod.h"

// test headers
#include "common/cpu_test2.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 0;
constexpr int ulps_double = 0;

constexpr auto ref_fmod = [](mpfr_double y, mpfr_double x) { return mpfr::fmod(y, x); };
constexpr auto ref_fmodf = [](mpfr_single y, mpfr_single x) { return mpfr::fmod(y, x); };

TEST_CASE("xtd::fmod", "[fmod][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::fmod(float, float)") {
    test_2<float, float, xtd::fmod, ref_fmod>(values, ulps_float);
  }

  SECTION("double xtd::fmod(double, double)") {
    test_2<double, double, xtd::fmod, ref_fmod>(values, ulps_double);
  }

  SECTION("double xtd::fmod(int, int)") {
    test_2<double, int, xtd::fmod, ref_fmod>(values, ulps_double);
  }

  SECTION("float xtd::fmodf(float, float)") {
    test_2f<float, float, xtd::fmodf, ref_fmodf>(values, ulps_float);
  }

  SECTION("float xtd::fmodf(double, double)") {
    test_2f<float, double, xtd::fmodf, ref_fmodf>(values, ulps_float);
  }

  SECTION("float xtd::fmodf(int, int)") {
    test_2f<float, int, xtd::fmodf, ref_fmodf>(values, ulps_float);
  }
}
