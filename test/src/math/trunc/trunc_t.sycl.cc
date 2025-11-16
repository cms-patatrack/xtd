/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/trunc.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_trunc.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::trunc", "[trunc][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::trunc(float)") {
            validate<float, float, xtd::trunc, mpfr_truncf>(platform, device, ulps_single);
          }

          SECTION("double xtd::trunc(double)") {
            validate<double, double, xtd::trunc, mpfr_trunc>(platform, device, ulps_double);
          }

          SECTION("double xtd::trunc(int)") {
            validate<double, int, xtd::trunc, mpfr_trunc>(platform, device, ulps_double);
          }

          SECTION("float xtd::truncf(float)") {
            validate<float, float, xtd::truncf, mpfr_truncf>(platform, device, ulps_single);
          }

          SECTION("float xtd::truncf(double)") {
            validate<float, double, xtd::truncf, mpfr_truncf>(platform, device, ulps_single);
          }

          SECTION("float xtd::truncf(int)") {
            validate<float, int, xtd::truncf, mpfr_truncf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
