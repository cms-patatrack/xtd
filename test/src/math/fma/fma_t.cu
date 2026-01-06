/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/fma.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "mpfr_fma.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::fma", "[fma][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::fma(float, float, float)") {
          validate<float, float, xtd::fma, mpfr_fmaf>(device, ulps_single);
        }

        SECTION("double xtd::fma(double, double, double)") {
          validate<double, double, xtd::fma, mpfr_fma>(device, ulps_double);
        }

        SECTION("double xtd::fma(int, int, int)") {
          validate<double, int, xtd::fma, mpfr_fma>(device, ulps_double);
        }

        SECTION("float xtd::fmaf(float, float, float)") {
          validate<float, float, xtd::fmaf, mpfr_fmaf>(device, ulps_single);
        }

        SECTION("float xtd::fmaf(double, double, double)") {
          validate<float, double, xtd::fmaf, mpfr_fmaf>(device, ulps_single);
        }

        SECTION("float xtd::fmaf(int, int, int)") {
          validate<float, int, xtd::fmaf, mpfr_fmaf>(device, ulps_single);
        }
      }
    }
  }
}
