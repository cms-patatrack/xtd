/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/acos.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "mpfr_acos.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::acos", "[acos][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::acos(float)") {
          validate<float, float, xtd::acos, mpfr_acosf>(device, ulps_single);
        }

        SECTION("double xtd::acos(double)") {
          validate<double, double, xtd::acos, mpfr_acos>(device, ulps_double);
        }

        SECTION("double xtd::acos(int)") {
          validate<double, int, xtd::acos, mpfr_acos>(device, ulps_double);
        }

        SECTION("float xtd::acosf(float)") {
          validate<float, float, xtd::acosf, mpfr_acosf>(device, ulps_single);
        }

        SECTION("float xtd::acosf(double)") {
          validate<float, double, xtd::acosf, mpfr_acosf>(device, ulps_single);
        }

        SECTION("float xtd::acosf(int)") {
          validate<float, int, xtd::acosf, mpfr_acosf>(device, ulps_single);
        }
      }
    }
  }
}
