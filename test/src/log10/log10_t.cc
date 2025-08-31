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

constexpr int ulps_float = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::log10", "[log10][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::log10(float)") {
    test<float, float, xtd::log10, mpfr::log10>(values, ulps_float);
  }

  SECTION("double xtd::log10(double)") {
    test<double, double, xtd::log10, mpfr::log10>(values, ulps_double);
  }

  SECTION("double xtd::log10(int)") {
    test<double, int, xtd::log10, mpfr::log10>(values, ulps_double);
  }

  SECTION("float xtd::log10f(float)") {
    test_f<float, float, xtd::log10f, mpfr::log10>(values, ulps_float);
  }

  SECTION("float xtd::log10f(double)") {
    test_f<float, double, xtd::log10f, mpfr::log10>(values, ulps_float);
  }

  SECTION("float xtd::log10f(int)") {
    test_f<float, int, xtd::log10f, mpfr::log10>(values, ulps_float);
  }
}
