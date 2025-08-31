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
#include "xtd/math/cos.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::cos", "[cos][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::cos(float)") {
    test<float, float, xtd::cos, mpfr::cos>(values, ulps_float);
  }

  SECTION("double xtd::cos(double)") {
    test<double, double, xtd::cos, mpfr::cos>(values, ulps_double);
  }

  SECTION("double xtd::cos(int)") {
    test<double, int, xtd::cos, mpfr::cos>(values, ulps_double);
  }

  SECTION("float xtd::cosf(float)") {
    test_f<float, float, xtd::cosf, mpfr::cos>(values, ulps_float);
  }

  SECTION("float xtd::cosf(double)") {
    test_f<float, double, xtd::cosf, mpfr::cos>(values, ulps_float);
  }

  SECTION("float xtd::cosf(int)") {
    test_f<float, int, xtd::cosf, mpfr::cos>(values, ulps_float);
  }
}
