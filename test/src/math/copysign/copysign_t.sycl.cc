/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/copysign.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "reference_copysign.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::copysign", "[copysign][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::copysign(float, float)") {
            test::sycl::validate<float, float, xtd::copysign, reference_copysignf>(platform, device, ulps_single);
          }

          SECTION("double xtd::copysign(double, double)") {
            test::sycl::validate<double, double, xtd::copysign, reference_copysign>(platform, device, ulps_double);
          }

          SECTION("double xtd::copysign(int, int)") {
            test::sycl::validate<double, int, xtd::copysign, reference_copysign>(platform, device, ulps_double);
          }

          SECTION("float xtd::copysignf(float, float)") {
            test::sycl::validate<float, float, xtd::copysignf, reference_copysignf>(platform, device, ulps_single);
          }

          SECTION("float xtd::copysignf(double, double)") {
            test::sycl::validate<float, double, xtd::copysignf, reference_copysignf>(platform, device, ulps_single);
          }

          SECTION("float xtd::copysignf(int, int)") {
            test::sycl::validate<float, int, xtd::copysignf, reference_copysignf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
