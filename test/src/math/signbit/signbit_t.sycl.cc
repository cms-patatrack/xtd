/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/signbit.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "reference_signbit.h"

TEST_CASE("xtd::signbit", "[signbit][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("int xtd::signbit(float)") {
            validate<int, float, xtd::signbit, reference_signbit>(platform, device);
          }

          SECTION("int xtd::signbit(double)") {
            validate<int, double, xtd::signbit, reference_signbit>(platform, device);
          }

          SECTION("int xtd::signbit(int)") {
            validate<int, int, xtd::signbit, reference_signbit>(platform, device);
          }

          SECTION("int xtd::signbit(unsigned int)") {
            validate<int, unsigned int, xtd::signbit, reference_signbit>(platform, device);
          }
        }
      }
    }
  }
}
