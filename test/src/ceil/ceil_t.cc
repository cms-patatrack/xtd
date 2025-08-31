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
#include "xtd/math/ceil.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::ceil", "[ceil][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::ceil(float)") {
    test<float, float, xtd::ceil, mpfr::ceil>(values, ulps_float);
  }

  SECTION("double xtd::ceil(double)") {
    test<double, double, xtd::ceil, mpfr::ceil>(values, ulps_double);
  }

  SECTION("double xtd::ceil(int)") {
    test<double, int, xtd::ceil, mpfr::ceil>(values, ulps_double);
  }

  SECTION("float xtd::ceilf(float)") {
    test_f<float, float, xtd::ceilf, mpfr::ceil>(values, ulps_float);
  }

  SECTION("float xtd::ceilf(double)") {
    test_f<float, double, xtd::ceilf, mpfr::ceil>(values, ulps_float);
  }

  SECTION("float xtd::ceilf(int)") {
    test_f<float, int, xtd::ceilf, mpfr::ceil>(values, ulps_float);
  }
}
