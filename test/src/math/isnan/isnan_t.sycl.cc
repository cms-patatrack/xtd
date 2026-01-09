/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/isnan.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "reference_isnan.h"

TEST_CASE("xtd::isnan", "[isnan][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("int xtd::isnan(float)") {
            validate<int, float, xtd::isnan, reference_isnan>(platform, device);
          }

          SECTION("int xtd::isnan(double)") {
            validate<int, double, xtd::isnan, reference_isnan>(platform, device);
          }

          SECTION("int xtd::isnan(int)") {
            validate<int, int, xtd::isnan, reference_isnan>(platform, device);
          }
        }
      }
    }
  }
}
