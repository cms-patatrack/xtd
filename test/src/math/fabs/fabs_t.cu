/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/fabs.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "mpfr_fabs.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::fabs", "[fabs][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::fabs(float)") {
          validate<float, float, xtd::fabs, mpfr_fabsf>(device, ulps_single);
        }

        SECTION("double xtd::fabs(double)") {
          validate<double, double, xtd::fabs, mpfr_fabs>(device, ulps_double);
        }

        SECTION("double xtd::fabs(int)") {
          validate<double, int, xtd::fabs, mpfr_fabs>(device, ulps_double);
        }

        SECTION("float xtd::fabsf(float)") {
          validate<float, float, xtd::fabsf, mpfr_fabsf>(device, ulps_single);
        }

        SECTION("float xtd::fabsf(double)") {
          validate<float, double, xtd::fabsf, mpfr_fabsf>(device, ulps_single);
        }

        SECTION("float xtd::fabsf(int)") {
          validate<float, int, xtd::fabsf, mpfr_fabsf>(device, ulps_single);
        }
      }
    }
  }
}
