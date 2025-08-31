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
#include "xtd/math/asin.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::asin", "[asin][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::asin(float)") {
    test<float, float, xtd::asin, mpfr::asin>(values, ulps_float);
  }

  SECTION("double xtd::asin(double)") {
    test<double, double, xtd::asin, mpfr::asin>(values, ulps_double);
  }

  SECTION("double xtd::asin(int)") {
    test<double, int, xtd::asin, mpfr::asin>(values, ulps_double);
  }

  SECTION("float xtd::asinf(float)") {
    test_f<float, float, xtd::asinf, mpfr::asin>(values, ulps_float);
  }

  SECTION("float xtd::asinf(double)") {
    test_f<float, double, xtd::asinf, mpfr::asin>(values, ulps_float);
  }

  SECTION("float xtd::asinf(int)") {
    test_f<float, int, xtd::asinf, mpfr::asin>(values, ulps_float);
  }
}
