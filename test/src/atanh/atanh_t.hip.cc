/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/atanh.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "mpfr_atanh.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::atanh", "[atanh][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::atanh(float)") {
          validate<float, float, xtd::atanh, mpfr_atanhf>(device, ulps_single);
        }

        SECTION("double xtd::atanh(double)") {
          validate<double, double, xtd::atanh, mpfr_atanh>(device, ulps_double);
        }

        SECTION("double xtd::atanh(int)") {
          validate<double, int, xtd::atanh, mpfr_atanh>(device, ulps_double);
        }

        SECTION("float xtd::atanhf(float)") {
          validate<float, float, xtd::atanhf, mpfr_atanhf>(device, ulps_single);
        }

        SECTION("float xtd::atanhf(double)") {
          validate<float, double, xtd::atanhf, mpfr_atanhf>(device, ulps_single);
        }

        SECTION("float xtd::atanhf(int)") {
          validate<float, int, xtd::atanhf, mpfr_atanhf>(device, ulps_single);
        }
      }
    }
  }
}
