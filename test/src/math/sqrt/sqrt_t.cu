/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/sqrt.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "mpfr_sqrt.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::sqrt", "[sqrt][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::sqrt(float)") {
          validate<float, float, xtd::sqrt, mpfr_sqrtf>(device, ulps_single);
        }

        SECTION("double xtd::sqrt(double)") {
          validate<double, double, xtd::sqrt, mpfr_sqrt>(device, ulps_double);
        }

        SECTION("double xtd::sqrt(int)") {
          validate<double, int, xtd::sqrt, mpfr_sqrt>(device, ulps_double);
        }

        SECTION("float xtd::sqrtf(float)") {
          validate<float, float, xtd::sqrtf, mpfr_sqrtf>(device, ulps_single);
        }

        SECTION("float xtd::sqrtf(double)") {
          validate<float, double, xtd::sqrtf, mpfr_sqrtf>(device, ulps_single);
        }

        SECTION("float xtd::sqrtf(int)") {
          validate<float, int, xtd::sqrtf, mpfr_sqrtf>(device, ulps_single);
        }
      }
    }
  }
}
