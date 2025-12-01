/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/round.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_round.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::round", "[round][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::round(float)") {
            validate<float, float, xtd::round, mpfr_roundf>(platform, device, ulps_single);
          }

          SECTION("double xtd::round(double)") {
            validate<double, double, xtd::round, mpfr_round>(platform, device, ulps_double);
          }

          SECTION("double xtd::round(int)") {
            validate<double, int, xtd::round, mpfr_round>(platform, device, ulps_double);
          }

          SECTION("float xtd::roundf(float)") {
            validate<float, float, xtd::roundf, mpfr_roundf>(platform, device, ulps_single);
          }

          SECTION("float xtd::roundf(double)") {
            validate<float, double, xtd::roundf, mpfr_roundf>(platform, device, ulps_single);
          }

          SECTION("float xtd::roundf(int)") {
            validate<float, int, xtd::roundf, mpfr_roundf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
