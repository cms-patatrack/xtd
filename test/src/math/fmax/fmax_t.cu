/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/fmax.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "mpfr_fmax.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::fmax", "[fmax][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::fmax(float, float)") {
          validate<float, float, xtd::fmax, mpfr_fmaxf>(device, ulps_single);
        }

        SECTION("double xtd::fmax(double, double)") {
          validate<double, double, xtd::fmax, mpfr_fmax>(device, ulps_double);
        }

        SECTION("double xtd::fmax(int, int)") {
          validate<double, int, xtd::fmax, mpfr_fmax>(device, ulps_double);
        }

        SECTION("float xtd::fmaxf(float, float)") {
          validate<float, float, xtd::fmaxf, mpfr_fmaxf>(device, ulps_single);
        }

        SECTION("float xtd::fmaxf(double, double)") {
          validate<float, double, xtd::fmaxf, mpfr_fmaxf>(device, ulps_single);
        }

        SECTION("float xtd::fmaxf(int, int)") {
          validate<float, int, xtd::fmaxf, mpfr_fmaxf>(device, ulps_single);
        }
      }
    }
  }
}
