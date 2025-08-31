/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cmath>
#include <iostream>
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
#include "xtd/math/expm1.h"

// test headers
#include "common/hip_check.h"
#include "common/hip_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 1;
constexpr int ulps_double = 2;  // 1 ULP for x ∈ [-10., +10.] according to the documentation

TEST_CASE("xtd::expm1", "[expm1][hip]") {
  std::vector<double> values = generate_input_values();

  int deviceCount;
  hipError_t hipStatus = hipGetDeviceCount(&deviceCount);

  if (hipStatus != hipSuccess || deviceCount == 0) {
    std::cout << "No AMD GPUs found, the test will be skipped.\n\n";
    exit(EXIT_SUCCESS);
  }

  for (int device = 0; device < deviceCount; ++device) {
    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, device));
    std::string section = "HIP GPU "s + std::to_string(device) + ": "s + properties.name;
    SECTION(section) {
      // set the current GPU
      HIP_CHECK(hipSetDevice(device));

      // create a HIP stream for all the asynchronous operations on this GPU
      hipStream_t queue;
      HIP_CHECK(hipStreamCreate(&queue));

      SECTION("float xtd::expm1(float)") {
        test<float, float, xtd::expm1, mpfr::expm1>(queue, values, ulps_float);
      }

      SECTION("double xtd::expm1(double)") {
        test<double, double, xtd::expm1, mpfr::expm1>(queue, values, ulps_double);
      }

      SECTION("double xtd::expm1(int)") {
        test<double, int, xtd::expm1, mpfr::expm1>(queue, values, ulps_double);
      }

      SECTION("float xtd::expm1f(float)") {
        test_f<float, float, xtd::expm1f, mpfr::expm1>(queue, values, ulps_float);
      }

      SECTION("float xtd::expm1f(double)") {
        test_f<float, double, xtd::expm1f, mpfr::expm1>(queue, values, ulps_float);
      }

      SECTION("float xtd::expm1f(int)") {
        test_f<float, int, xtd::expm1f, mpfr::expm1>(queue, values, ulps_float);
      }

      HIP_CHECK(hipStreamDestroy(queue));
    }
  }
}
