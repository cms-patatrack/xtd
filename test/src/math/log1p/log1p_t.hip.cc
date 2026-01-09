/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/log1p.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "mpfr_log1p.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::log1p", "[log1p][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::log1p(float)") {
          validate<float, float, xtd::log1p, mpfr_log1pf>(device, ulps_single);
        }

        SECTION("double xtd::log1p(double)") {
          validate<double, double, xtd::log1p, mpfr_log1p>(device, ulps_double);
        }

        SECTION("double xtd::log1p(int)") {
          validate<double, int, xtd::log1p, mpfr_log1p>(device, ulps_double);
        }

        SECTION("float xtd::log1pf(float)") {
          validate<float, float, xtd::log1pf, mpfr_log1pf>(device, ulps_single);
        }

        SECTION("float xtd::log1pf(double)") {
          validate<float, double, xtd::log1pf, mpfr_log1pf>(device, ulps_single);
        }

        SECTION("float xtd::log1pf(int)") {
          validate<float, int, xtd::log1pf, mpfr_log1pf>(device, ulps_single);
        }
      }
    }
  }
}
