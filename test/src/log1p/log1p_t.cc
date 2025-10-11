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
#include "xtd/math/log1p.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

constexpr auto ref_function = [](mpfr_double x) { return mpfr::log1p(x); };
constexpr auto ref_functionf = [](mpfr_single x) { return mpfr::log1p(x); };

TEST_CASE("xtd::log1p", "[log1p][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::log1p(float)") {
    test_a<float, float, xtd::log1p, ref_function>(values, ulps_single);
  }

  SECTION("double xtd::log1p(double)") {
    test_a<double, double, xtd::log1p, ref_function>(values, ulps_double);
  }

  SECTION("double xtd::log1p(int)") {
    test_a<double, int, xtd::log1p, ref_function>(values, ulps_double);
  }

  SECTION("float xtd::log1pf(float)") {
    test_f<float, float, xtd::log1pf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::log1pf(double)") {
    test_f<float, double, xtd::log1pf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::log1pf(int)") {
    test_f<float, int, xtd::log1pf, ref_functionf>(values, ulps_single);
  }
}
