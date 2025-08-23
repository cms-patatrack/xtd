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
#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// HIP headers
#include <hip/hip_runtime.h>

// xtd headers
#include "xtd/math/sin.h"

// test headers
#include "common/hip_check.h"
#include "common/hip_test.h"
#include "common/math_inputs.h"

TEST_CASE("xtd::sin", "[sin][hip]") {
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

      SECTION("float xtd::sin(float)") {
        test<float, float, xtd::sin, std::sin>(queue, values);
      }

      SECTION("double xtd::sin(double)") {
        test<double, double, xtd::sin, std::sin>(queue, values);
      }

      SECTION("double xtd::sin(int)") {
        // Note: HIP/ROCm does not provide the std::sin() overload for integer arguments.
        test<double, int, xtd::sin, [](int arg) { return std::sin(static_cast<double>(arg)); }>(queue, values);
      }

      SECTION("float xtd::sinf(float)") {
        test_f<float, float, xtd::sinf, std::sinf>(queue, values);
      }

      SECTION("float xtd::sinf(double)") {
        test_f<float, double, xtd::sinf, std::sinf>(queue, values);
      }

      SECTION("float xtd::sinf(int)") {
        test_f<float, int, xtd::sinf, std::sinf>(queue, values);
      }

      HIP_CHECK(hipStreamDestroy(queue));
    }
  }
}
