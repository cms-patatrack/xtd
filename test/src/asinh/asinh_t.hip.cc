/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <string>
#include <vector>
using namespace std::literals;

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// HIP headers
#include <hip/hip_runtime.h>

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include "xtd/math/asinh.h"

// test headers
#include "common/hip_check.h"
#include "common/hip_test.h"
#include "common/hip_version.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

constexpr auto ref_function = [](mpfr_double x) { return mpfr::asinh(x); };
constexpr auto ref_functionf = [](mpfr_single x) { return mpfr::asinh(x); };

TEST_CASE("xtd::asinh", "[asinh][hip]") {
  std::vector<double> values = generate_input_values();

  DYNAMIC_SECTION("HIP platform: " << hip_version()) {
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    for (int device = 0; device < deviceCount; ++device) {
      hipDeviceProp_t properties;
      HIP_CHECK(hipGetDeviceProperties(&properties, device));
      DYNAMIC_SECTION("HIP device " << device << ": " << properties.name) {
        // set the current GPU
        HIP_CHECK(hipSetDevice(device));

        // create a HIP stream for all the asynchronous operations on this GPU
        hipStream_t queue;
        HIP_CHECK(hipStreamCreate(&queue));

        SECTION("float xtd::asinh(float)") {
          test_a<float, float, xtd::asinh, ref_function>(queue, values, ulps_single);
        }

        SECTION("double xtd::asinh(double)") {
          test_a<double, double, xtd::asinh, ref_function>(queue, values, ulps_double);
        }

        SECTION("double xtd::asinh(int)") {
          test_a<double, int, xtd::asinh, ref_function>(queue, values, ulps_double);
        }

        SECTION("float xtd::asinhf(float)") {
          test_f<float, float, xtd::asinhf, ref_functionf>(queue, values, ulps_single);
        }

        SECTION("float xtd::asinhf(double)") {
          test_f<float, double, xtd::asinhf, ref_functionf>(queue, values, ulps_single);
        }

        SECTION("float xtd::asinhf(int)") {
          test_f<float, int, xtd::asinhf, ref_functionf>(queue, values, ulps_single);
        }

        HIP_CHECK(hipStreamDestroy(queue));
      }
    }
  }
}
