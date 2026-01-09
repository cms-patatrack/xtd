/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/ceil.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_ceil.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::ceil", "[ceil][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::ceil(float)") {
            validate<float, float, xtd::ceil, mpfr_ceilf>(platform, device, ulps_single);
          }

          SECTION("double xtd::ceil(double)") {
            validate<double, double, xtd::ceil, mpfr_ceil>(platform, device, ulps_double);
          }

          SECTION("double xtd::ceil(int)") {
            validate<double, int, xtd::ceil, mpfr_ceil>(platform, device, ulps_double);
          }

          SECTION("float xtd::ceilf(float)") {
            validate<float, float, xtd::ceilf, mpfr_ceilf>(platform, device, ulps_single);
          }

          SECTION("float xtd::ceilf(double)") {
            validate<float, double, xtd::ceilf, mpfr_ceilf>(platform, device, ulps_single);
          }

          SECTION("float xtd::ceilf(int)") {
            validate<float, int, xtd::ceilf, mpfr_ceilf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
