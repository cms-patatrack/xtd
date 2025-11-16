/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/hypot.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "mpfr_hypot.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::hypot", "[hypot][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::hypot(float, float)") {
          validate<float, float, xtd::hypot, mpfr_hypotf>(device, ulps_single);
        }

        SECTION("double xtd::hypot(double, double)") {
          validate<double, double, xtd::hypot, mpfr_hypot>(device, ulps_double);
        }

        SECTION("double xtd::hypot(int, int)") {
          validate<double, int, xtd::hypot, mpfr_hypot>(device, ulps_double);
        }

        SECTION("float xtd::hypotf(float, float)") {
          validate<float, float, xtd::hypotf, mpfr_hypotf>(device, ulps_single);
        }

        SECTION("float xtd::hypotf(double, double)") {
          validate<float, double, xtd::hypotf, mpfr_hypotf>(device, ulps_single);
        }

        SECTION("float xtd::hypotf(int, int)") {
          validate<float, int, xtd::hypotf, mpfr_hypotf>(device, ulps_single);
        }
      }
    }
  }
}
