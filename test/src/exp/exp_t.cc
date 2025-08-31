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
#include "xtd/math/exp.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::exp", "[exp][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::exp(float)") {
    test<float, float, xtd::exp, mpfr::exp>(values, ulps_float);
  }

  SECTION("double xtd::exp(double)") {
    test<double, double, xtd::exp, mpfr::exp>(values, ulps_double);
  }

  SECTION("double xtd::exp(int)") {
    test<double, int, xtd::exp, mpfr::exp>(values, ulps_double);
  }

  SECTION("float xtd::expf(float)") {
    test_f<float, float, xtd::expf, mpfr::exp>(values, ulps_float);
  }

  SECTION("float xtd::expf(double)") {
    test_f<float, double, xtd::expf, mpfr::exp>(values, ulps_float);
  }

  SECTION("float xtd::expf(int)") {
    test_f<float, int, xtd::expf, mpfr::exp>(values, ulps_float);
  }
}
