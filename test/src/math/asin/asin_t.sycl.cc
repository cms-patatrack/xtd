/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/asin.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_asin.h"

constexpr int ulps_single = 4;
constexpr int ulps_double = 4;

TEST_CASE("xtd::asin", "[asin][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::asin(float)") {
            validate<float, float, xtd::asin, mpfr_asinf>(platform, device, ulps_single);
          }

          SECTION("double xtd::asin(double)") {
            validate<double, double, xtd::asin, mpfr_asin>(platform, device, ulps_double);
          }

          SECTION("double xtd::asin(int)") {
            validate<double, int, xtd::asin, mpfr_asin>(platform, device, ulps_double);
          }

          SECTION("float xtd::asinf(float)") {
            validate<float, float, xtd::asinf, mpfr_asinf>(platform, device, ulps_single);
          }

          SECTION("float xtd::asinf(double)") {
            validate<float, double, xtd::asinf, mpfr_asinf>(platform, device, ulps_single);
          }

          SECTION("float xtd::asinf(int)") {
            validate<float, int, xtd::asinf, mpfr_asinf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
