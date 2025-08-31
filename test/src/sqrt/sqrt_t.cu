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

// CUDA headers
#include <cuda_runtime.h>

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include "xtd/math/sqrt.h"

// test headers
#include "common/cuda_check.h"
#include "common/cuda_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::sqrt", "[sqrt][cuda]") {
  std::vector<double> values = generate_input_values();

  int deviceCount;
  cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

  if (cudaStatus != cudaSuccess || deviceCount == 0) {
    std::cout << "No NVIDIA GPUs found, the test will be skipped.\n\n";
    exit(EXIT_SUCCESS);
  }

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

      SECTION("float xtd::sqrt(float)") {
        test<float, float, xtd::sqrt, mpfr::sqrt>(queue, values, ulps_float);
      }

      SECTION("double xtd::sqrt(double)") {
        test<double, double, xtd::sqrt, mpfr::sqrt>(queue, values, ulps_double);
      }

      SECTION("double xtd::sqrt(int)") {
        test<double, int, xtd::sqrt, mpfr::sqrt>(queue, values, ulps_double);
      }

      SECTION("float xtd::sqrtf(float)") {
        test_f<float, float, xtd::sqrtf, mpfr::sqrt>(queue, values, ulps_float);
      }

      SECTION("float xtd::sqrtf(double)") {
        test_f<float, double, xtd::sqrtf, mpfr::sqrt>(queue, values, ulps_float);
      }

      SECTION("float xtd::sqrtf(int)") {
        test_f<float, int, xtd::sqrtf, mpfr::sqrt>(queue, values, ulps_float);
      }

      CUDA_CHECK(cudaStreamDestroy(queue));
    }
  }
}
