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

// CUDA headers
#include <cuda_runtime.h>

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include "xtd/math/atanh.h"

// test headers
#include "common/cuda_check.h"
#include "common/cuda_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 3;
constexpr int ulps_double = 2;

TEST_CASE("xtd::atanh", "[atanh][cuda]") {
  std::vector<double> values = generate_input_values();

  int deviceCount;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

  for (int device = 0; device < deviceCount; ++device) {
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    std::string section = "CUDA GPU "s + std::to_string(device) + ": "s + properties.name;
    SECTION(section) {
      // set the current GPU
      CUDA_CHECK(cudaSetDevice(device));

      // create a CUDA stream for all the asynchronous operations on this GPU
      cudaStream_t queue;
      CUDA_CHECK(cudaStreamCreate(&queue));

      SECTION("float xtd::atanh(float)") {
        test<float, float, xtd::atanh, mpfr::atanh>(queue, values, ulps_float);
      }

      SECTION("double xtd::atanh(double)") {
        test<double, double, xtd::atanh, mpfr::atanh>(queue, values, ulps_double);
      }

      SECTION("double xtd::atanh(int)") {
        test<double, int, xtd::atanh, mpfr::atanh>(queue, values, ulps_double);
      }

      SECTION("float xtd::atanhf(float)") {
        test_f<float, float, xtd::atanhf, mpfr::atanh>(queue, values, ulps_float);
      }

      SECTION("float xtd::atanhf(double)") {
        test_f<float, double, xtd::atanhf, mpfr::atanh>(queue, values, ulps_float);
      }

      SECTION("float xtd::atanhf(int)") {
        test_f<float, int, xtd::atanhf, mpfr::atanh>(queue, values, ulps_float);
      }

      CUDA_CHECK(cudaStreamDestroy(queue));
    }
  }
}
