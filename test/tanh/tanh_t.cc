/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cmath>
#include <vector>

// Catch2 headers
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include "xtd/math/tanh.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::tanh", "[tanh][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::tanh(float)") {
    test<float, float, xtd::tanh, mpfr::tanh>(values, ulps_float);
  }

  SECTION("double xtd::tanh(double)") {
    test<double, double, xtd::tanh, mpfr::tanh>(values, ulps_double);
  }

  SECTION("double xtd::tanh(int)") {
    test<double, int, xtd::tanh, mpfr::tanh>(values, ulps_double);
  }

  SECTION("float xtd::tanhf(float)") {
    test_f<float, float, xtd::tanhf, mpfr::tanh>(values, ulps_float);
  }

  SECTION("float xtd::tanhf(double)") {
    test_f<float, double, xtd::tanhf, mpfr::tanh>(values, ulps_float);
  }

  SECTION("float xtd::tanhf(int)") {
    test_f<float, int, xtd::tanhf, mpfr::tanh>(values, ulps_float);
  }
}
