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

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include "xtd/math/pow.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 1;
constexpr int ulps_double = 1;

constexpr auto ref_pow = [](mpfr_double y, mpfr_double x) { return mpfr::pow(y, x); };
constexpr auto ref_powf = [](mpfr_single y, mpfr_single x) { return mpfr::pow(y, x); };

TEST_CASE("xtd::pow", "[pow][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::pow(float, float)") {
    test_2<float, float, xtd::pow, ref_pow>(values, ulps_float);
  }

  SECTION("double xtd::pow(double, double)") {
    test_2<double, double, xtd::pow, ref_pow>(values, ulps_double);
  }

  SECTION("double xtd::pow(int, int)") {
    test_2<double, int, xtd::pow, ref_pow>(values, ulps_double);
  }

  SECTION("float xtd::powf(float, float)") {
    test_2f<float, float, xtd::powf, ref_powf>(values, ulps_float);
  }

  SECTION("float xtd::powf(double, double)") {
    test_2f<float, double, xtd::powf, ref_powf>(values, ulps_float);
  }

  SECTION("float xtd::powf(int, int)") {
    test_2f<float, int, xtd::powf, ref_powf>(values, ulps_float);
  }
}
