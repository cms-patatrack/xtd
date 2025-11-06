/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/tan.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "mpfr_tan.h"

constexpr int ulps_single = 4;
constexpr int ulps_double = 2;

TEST_CASE("xtd::tan", "[tan][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::tan(float)") {
          validate<float, float, xtd::tan, mpfr_tanf>(device, ulps_single);
        }

        SECTION("double xtd::tan(double)") {
          validate<double, double, xtd::tan, mpfr_tan>(device, ulps_double);
        }

        SECTION("double xtd::tan(int)") {
          validate<double, int, xtd::tan, mpfr_tan>(device, ulps_double);
        }

        SECTION("float xtd::tanf(float)") {
          validate<float, float, xtd::tanf, mpfr_tanf>(device, ulps_single);
        }

        SECTION("float xtd::tanf(double)") {
          validate<float, double, xtd::tanf, mpfr_tanf>(device, ulps_single);
        }

        SECTION("float xtd::tanf(int)") {
          validate<float, int, xtd::tanf, mpfr_tanf>(device, ulps_single);
        }
      }
    }
  }
}
