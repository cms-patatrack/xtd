/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cmath>
#include <vector>

// Catch2 headers
#include <catch.hpp>

// xtd headers
#include "xtd/math/tanh.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_tanh.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::tanh", "[tanh][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::tanh(float)") {
      validate<float, float, xtd::tanh, mpfr_tanhf>(device, ulps_single);
    }

    SECTION("double xtd::tanh(double)") {
      validate<double, double, xtd::tanh, mpfr_tanh>(device, ulps_double);
    }

    SECTION("double xtd::tanh(int)") {
      validate<double, int, xtd::tanh, mpfr_tanh>(device, ulps_double);
    }

    SECTION("float xtd::tanhf(float)") {
      validate<float, float, xtd::tanhf, mpfr_tanhf>(device, ulps_single);
    }

    SECTION("float xtd::tanhf(double)") {
      validate<float, double, xtd::tanhf, mpfr_tanhf>(device, ulps_single);
    }

    SECTION("float xtd::tanhf(int)") {
      validate<float, int, xtd::tanhf, mpfr_tanhf>(device, ulps_single);
    }
  }
}
