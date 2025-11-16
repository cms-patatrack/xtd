/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/atan2.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "mpfr_atan2.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::atan2", "[atan2][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::atan2(float, float)") {
          validate<float, float, xtd::atan2, mpfr_atan2f>(device, ulps_single);
        }

        SECTION("double xtd::atan2(double, double)") {
          validate<double, double, xtd::atan2, mpfr_atan2>(device, ulps_double);
        }

        SECTION("double xtd::atan2(int, int)") {
          validate<double, int, xtd::atan2, mpfr_atan2>(device, ulps_double);
        }

        SECTION("float xtd::atan2f(float, float)") {
          validate<float, float, xtd::atan2f, mpfr_atan2f>(device, ulps_single);
        }

        SECTION("float xtd::atan2f(double, double)") {
          validate<float, double, xtd::atan2f, mpfr_atan2f>(device, ulps_single);
        }

        SECTION("float xtd::atan2f(int, int)") {
          validate<float, int, xtd::atan2f, mpfr_atan2f>(device, ulps_single);
        }
      }
    }
  }
}
