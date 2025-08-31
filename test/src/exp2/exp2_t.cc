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
#include "xtd/math/exp2.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::exp2", "[exp2][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::exp2(float)") {
    test<float, float, xtd::exp2, mpfr::exp2>(values, ulps_float);
  }

  SECTION("double xtd::exp2(double)") {
    test<double, double, xtd::exp2, mpfr::exp2>(values, ulps_double);
  }

  SECTION("double xtd::exp2(int)") {
    test<double, int, xtd::exp2, mpfr::exp2>(values, ulps_double);
  }

  SECTION("float xtd::exp2f(float)") {
    test_f<float, float, xtd::exp2f, mpfr::exp2>(values, ulps_float);
  }

  SECTION("float xtd::exp2f(double)") {
    test_f<float, double, xtd::exp2f, mpfr::exp2>(values, ulps_float);
  }

  SECTION("float xtd::exp2f(int)") {
    test_f<float, int, xtd::exp2f, mpfr::exp2>(values, ulps_float);
  }
}
