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
#include "xtd/math/fmin.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 0;
constexpr int ulps_double = 0;

constexpr auto ref_fmin = [](mpfr_double y, mpfr_double x) { return mpfr::fmin(y, x); };
constexpr auto ref_fminf = [](mpfr_single y, mpfr_single x) { return mpfr::fmin(y, x); };

TEST_CASE("xtd::fmin", "[fmin][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::fmin(float, float)") {
    test_2<float, float, xtd::fmin, ref_fmin>(values, ulps_float);
  }

  SECTION("double xtd::fmin(double, double)") {
    test_2<double, double, xtd::fmin, ref_fmin>(values, ulps_double);
  }

  SECTION("double xtd::fmin(int, int)") {
    test_2<double, int, xtd::fmin, ref_fmin>(values, ulps_double);
  }

  SECTION("float xtd::fminf(float, float)") {
    test_2f<float, float, xtd::fminf, ref_fminf>(values, ulps_float);
  }

  SECTION("float xtd::fminf(double, double)") {
    test_2f<float, double, xtd::fminf, ref_fminf>(values, ulps_float);
  }

  SECTION("float xtd::fminf(int, int)") {
    test_2f<float, int, xtd::fminf, ref_fminf>(values, ulps_float);
  }
}
