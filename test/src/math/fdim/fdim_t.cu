/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/fdim.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "mpfr_fdim.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::fdim", "[fdim][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::fdim(float, float)") {
          validate<float, float, xtd::fdim, mpfr_fdimf>(device, ulps_single);
        }

        SECTION("double xtd::fdim(double, double)") {
          validate<double, double, xtd::fdim, mpfr_fdim>(device, ulps_double);
        }

        SECTION("double xtd::fdim(int, int)") {
          validate<double, int, xtd::fdim, mpfr_fdim>(device, ulps_double);
        }

        SECTION("float xtd::fdimf(float, float)") {
          validate<float, float, xtd::fdimf, mpfr_fdimf>(device, ulps_single);
        }

        SECTION("float xtd::fdimf(double, double)") {
          validate<float, double, xtd::fdimf, mpfr_fdimf>(device, ulps_single);
        }

        SECTION("float xtd::fdimf(int, int)") {
          validate<float, int, xtd::fdimf, mpfr_fdimf>(device, ulps_single);
        }
      }
    }
  }
}
