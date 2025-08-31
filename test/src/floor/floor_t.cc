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
#include "xtd/math/floor.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::floor", "[floor][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::floor(float)") {
    test<float, float, xtd::floor, mpfr::floor>(values, ulps_float);
  }

  SECTION("double xtd::floor(double)") {
    test<double, double, xtd::floor, mpfr::floor>(values, ulps_double);
  }

  SECTION("double xtd::floor(int)") {
    test<double, int, xtd::floor, mpfr::floor>(values, ulps_double);
  }

  SECTION("float xtd::floorf(float)") {
    test_f<float, float, xtd::floorf, mpfr::floor>(values, ulps_float);
  }

  SECTION("float xtd::floorf(double)") {
    test_f<float, double, xtd::floorf, mpfr::floor>(values, ulps_float);
  }

  SECTION("float xtd::floorf(int)") {
    test_f<float, int, xtd::floorf, mpfr::floor>(values, ulps_float);
  }
}
