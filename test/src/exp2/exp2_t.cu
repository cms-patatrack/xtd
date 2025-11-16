/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/exp2.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "mpfr_exp2.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 1;

TEST_CASE("xtd::exp2", "[exp2][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::exp2(float)") {
          validate<float, float, xtd::exp2, mpfr_exp2f>(device, ulps_single);
        }

        SECTION("double xtd::exp2(double)") {
          validate<double, double, xtd::exp2, mpfr_exp2>(device, ulps_double);
        }

        SECTION("double xtd::exp2(int)") {
          validate<double, int, xtd::exp2, mpfr_exp2>(device, ulps_double);
        }

        SECTION("float xtd::exp2f(float)") {
          validate<float, float, xtd::exp2f, mpfr_exp2f>(device, ulps_single);
        }

        SECTION("float xtd::exp2f(double)") {
          validate<float, double, xtd::exp2f, mpfr_exp2f>(device, ulps_single);
        }

        SECTION("float xtd::exp2f(int)") {
          validate<float, int, xtd::exp2f, mpfr_exp2f>(device, ulps_single);
        }
      }
    }
  }
}
