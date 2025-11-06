/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/log.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "mpfr_log.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::log", "[log][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::log(float)") {
          validate<float, float, xtd::log, mpfr_logf>(device, ulps_single);
        }

        SECTION("double xtd::log(double)") {
          validate<double, double, xtd::log, mpfr_log>(device, ulps_double);
        }

        SECTION("double xtd::log(int)") {
          validate<double, int, xtd::log, mpfr_log>(device, ulps_double);
        }

        SECTION("float xtd::logf(float)") {
          validate<float, float, xtd::logf, mpfr_logf>(device, ulps_single);
        }

        SECTION("float xtd::logf(double)") {
          validate<float, double, xtd::logf, mpfr_logf>(device, ulps_single);
        }

        SECTION("float xtd::logf(int)") {
          validate<float, int, xtd::logf, mpfr_logf>(device, ulps_single);
        }
      }
    }
  }
}
