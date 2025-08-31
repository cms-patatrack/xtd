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
#include "xtd/math/log.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::log", "[log][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::log(float)") {
    test<float, float, xtd::log, mpfr::log>(values, ulps_float);
  }

  SECTION("double xtd::log(double)") {
    test<double, double, xtd::log, mpfr::log>(values, ulps_double);
  }

  SECTION("double xtd::log(int)") {
    test<double, int, xtd::log, mpfr::log>(values, ulps_double);
  }

  SECTION("float xtd::logf(float)") {
    test_f<float, float, xtd::logf, mpfr::log>(values, ulps_float);
  }

  SECTION("float xtd::logf(double)") {
    test_f<float, double, xtd::logf, mpfr::log>(values, ulps_float);
  }

  SECTION("float xtd::logf(int)") {
    test_f<float, int, xtd::logf, mpfr::log>(values, ulps_float);
  }
}
