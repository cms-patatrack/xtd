/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/cos.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "mpfr_cos.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::cos", "[cos][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::cos(float)") {
          validate<float, float, xtd::cos, mpfr_cosf>(device, ulps_single);
        }

        SECTION("double xtd::cos(double)") {
          validate<double, double, xtd::cos, mpfr_cos>(device, ulps_double);
        }

        SECTION("double xtd::cos(int)") {
          validate<double, int, xtd::cos, mpfr_cos>(device, ulps_double);
        }

        SECTION("float xtd::cosf(float)") {
          validate<float, float, xtd::cosf, mpfr_cosf>(device, ulps_single);
        }

        SECTION("float xtd::cosf(double)") {
          validate<float, double, xtd::cosf, mpfr_cosf>(device, ulps_single);
        }

        SECTION("float xtd::cosf(int)") {
          validate<float, int, xtd::cosf, mpfr_cosf>(device, ulps_single);
        }
      }
    }
  }
}
