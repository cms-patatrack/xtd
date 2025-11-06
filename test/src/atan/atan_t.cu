/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/atan.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "mpfr_atan.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::atan", "[atan][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::atan(float)") {
          validate<float, float, xtd::atan, mpfr_atanf>(device, ulps_single);
        }

        SECTION("double xtd::atan(double)") {
          validate<double, double, xtd::atan, mpfr_atan>(device, ulps_double);
        }

        SECTION("double xtd::atan(int)") {
          validate<double, int, xtd::atan, mpfr_atan>(device, ulps_double);
        }

        SECTION("float xtd::atanf(float)") {
          validate<float, float, xtd::atanf, mpfr_atanf>(device, ulps_single);
        }

        SECTION("float xtd::atanf(double)") {
          validate<float, double, xtd::atanf, mpfr_atanf>(device, ulps_single);
        }

        SECTION("float xtd::atanf(int)") {
          validate<float, int, xtd::atanf, mpfr_atanf>(device, ulps_single);
        }
      }
    }
  }
}
