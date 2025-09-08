/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <iostream>
#include <sstream>
#include <stdexcept>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

namespace internal {

  [[noreturn]] inline void __abort_on_cuda_error(const char* file,
                                                 int line,
                                                 const char* cmd,
                                                 const char* error,
                                                 const char* message,
                                                 const char* description = nullptr) {
    std::ostringstream out;
    out << "\n";
    out << file << ", line " << line << ":\n";
    out << "CUDA_CHECK(" << cmd << ");\n";
    out << error << ": " << message << "\n";
    if (description)
      out << description << "\n";

    throw std::runtime_error(out.str());
  }

  inline void __cuda_check(
      const char* file, int line, const char* cmd, CUresult result, const char* description = nullptr) {
    if (result == CUDA_SUCCESS)
      return;

    const char* error;
    const char* message;
    cuGetErrorName(result, &error);
    cuGetErrorString(result, &message);
    __abort_on_cuda_error(file, line, cmd, error, message, description);
  }

  inline void __cuda_check(
      const char* file, int line, const char* cmd, cudaError_t result, const char* description = nullptr) {
    if (result == cudaSuccess)
      return;

    const char* error = cudaGetErrorName(result);
    const char* message = cudaGetErrorString(result);
    __abort_on_cuda_error(file, line, cmd, error, message, description);
  }

}  // namespace internal

#define CUDA_CHECK(ARG, ...) (internal::__cuda_check(__FILE__, __LINE__, #ARG, (ARG), ##__VA_ARGS__))
