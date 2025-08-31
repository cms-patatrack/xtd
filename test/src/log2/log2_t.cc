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
#include "xtd/math/log2.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 1;
constexpr int ulps_double = 2;

TEST_CASE("xtd::log2", "[log2][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::log2(float)") {
    test<float, float, xtd::log2, mpfr::log2>(values, ulps_float);
  }

  SECTION("double xtd::log2(double)") {
    test<double, double, xtd::log2, mpfr::log2>(values, ulps_double);
  }

  SECTION("double xtd::log2(int)") {
    test<double, int, xtd::log2, mpfr::log2>(values, ulps_double);
  }

  SECTION("float xtd::log2f(float)") {
    test_f<float, float, xtd::log2f, mpfr::log2>(values, ulps_float);
  }

  SECTION("float xtd::log2f(double)") {
    test_f<float, double, xtd::log2f, mpfr::log2>(values, ulps_float);
  }

  SECTION("float xtd::log2f(int)") {
    test_f<float, int, xtd::log2f, mpfr::log2>(values, ulps_float);
  }
}
