/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/fmod.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "mpfr_fmod.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::fmod", "[fmod][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::fmod(float, float)") {
          validate<float, float, xtd::fmod, mpfr_fmodf>(device, ulps_single);
        }

        SECTION("double xtd::fmod(double, double)") {
          validate<double, double, xtd::fmod, mpfr_fmod>(device, ulps_double);
        }

        SECTION("double xtd::fmod(int, int)") {
          validate<double, int, xtd::fmod, mpfr_fmod>(device, ulps_double);
        }

        SECTION("float xtd::fmodf(float, float)") {
          validate<float, float, xtd::fmodf, mpfr_fmodf>(device, ulps_single);
        }

        SECTION("float xtd::fmodf(double, double)") {
          validate<float, double, xtd::fmodf, mpfr_fmodf>(device, ulps_single);
        }

        SECTION("float xtd::fmodf(int, int)") {
          validate<float, int, xtd::fmodf, mpfr_fmodf>(device, ulps_single);
        }
      }
    }
  }
}
