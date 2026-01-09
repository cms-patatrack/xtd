/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/fmin.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_fmin.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::fmin", "[fmin][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::fmin(float, float)") {
            test::sycl::validate<float, float, xtd::fmin, mpfr_fminf>(platform, device, ulps_single);
          }

          SECTION("double xtd::fmin(double, double)") {
            test::sycl::validate<double, double, xtd::fmin, mpfr_fmin>(platform, device, ulps_double);
          }

          SECTION("double xtd::fmin(int, int)") {
            test::sycl::validate<double, int, xtd::fmin, mpfr_fmin>(platform, device, ulps_double);
          }

          SECTION("float xtd::fminf(float, float)") {
            test::sycl::validate<float, float, xtd::fminf, mpfr_fminf>(platform, device, ulps_single);
          }

          SECTION("float xtd::fminf(double, double)") {
            test::sycl::validate<float, double, xtd::fminf, mpfr_fminf>(platform, device, ulps_single);
          }

          SECTION("float xtd::fminf(int, int)") {
            test::sycl::validate<float, int, xtd::fminf, mpfr_fminf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
