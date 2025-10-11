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
#include "xtd/math/log10.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

constexpr auto ref_function = [](mpfr_double x) { return mpfr::log10(x); };
constexpr auto ref_functionf = [](mpfr_single x) { return mpfr::log10(x); };

TEST_CASE("xtd::log10", "[log10][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::log10(float)") {
    test_a<float, float, xtd::log10, ref_function>(values, ulps_single);
  }

  SECTION("double xtd::log10(double)") {
    test_a<double, double, xtd::log10, ref_function>(values, ulps_double);
  }

  SECTION("double xtd::log10(int)") {
    test_a<double, int, xtd::log10, ref_function>(values, ulps_double);
  }

  SECTION("float xtd::log10f(float)") {
    test_f<float, float, xtd::log10f, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::log10f(double)") {
    test_f<float, double, xtd::log10f, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::log10f(int)") {
    test_f<float, int, xtd::log10f, ref_functionf>(values, ulps_single);
  }
}
