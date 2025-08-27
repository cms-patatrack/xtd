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
#include "xtd/math/remainder.h"

// test headers
#include "common/cpu_test2.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 0;
constexpr int ulps_double = 0;

constexpr auto ref_remainder = [](mpfr_double y, mpfr_double x) { return mpfr::remainder(y, x); };
constexpr auto ref_remainderf = [](mpfr_single y, mpfr_single x) { return mpfr::remainder(y, x); };

TEST_CASE("xtd::remainder", "[remainder][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::remainder(float, float)") {
    test_2<float, float, xtd::remainder, ref_remainder>(values, ulps_float);
  }

  SECTION("double xtd::remainder(double, double)") {
    test_2<double, double, xtd::remainder, ref_remainder>(values, ulps_double);
  }

  SECTION("double xtd::remainder(int, int)") {
    test_2<double, int, xtd::remainder, ref_remainder>(values, ulps_double);
  }

  SECTION("float xtd::remainderf(float, float)") {
    test_2f<float, float, xtd::remainderf, ref_remainderf>(values, ulps_float);
  }

  SECTION("float xtd::remainderf(double, double)") {
    test_2f<float, double, xtd::remainderf, ref_remainderf>(values, ulps_float);
  }

  SECTION("float xtd::remainderf(int, int)") {
    test_2f<float, int, xtd::remainderf, ref_remainderf>(values, ulps_float);
  }
}
