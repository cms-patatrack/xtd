/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/fdim.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_fdim.h"

constexpr int ulps_single = 1;  // Correctly rounded according to the OpenCL documentation.
constexpr int ulps_double = 0;

TEST_CASE("xtd::fdim", "[fdim][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::fdim(float, float)") {
            test::sycl::validate<float, float, xtd::fdim, mpfr_fdimf>(platform, device, ulps_single);
          }

          SECTION("double xtd::fdim(double, double)") {
            test::sycl::validate<double, double, xtd::fdim, mpfr_fdim>(platform, device, ulps_double);
          }

          SECTION("double xtd::fdim(int, int)") {
            test::sycl::validate<double, int, xtd::fdim, mpfr_fdim>(platform, device, ulps_double);
          }

          SECTION("float xtd::fdimf(float, float)") {
            test::sycl::validate<float, float, xtd::fdimf, mpfr_fdimf>(platform, device, ulps_single);
          }

          SECTION("float xtd::fdimf(double, double)") {
            test::sycl::validate<float, double, xtd::fdimf, mpfr_fdimf>(platform, device, ulps_single);
          }

          SECTION("float xtd::fdimf(int, int)") {
            test::sycl::validate<float, int, xtd::fdimf, mpfr_fdimf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
