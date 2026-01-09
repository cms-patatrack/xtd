/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <iostream>
#include <sstream>
#include <stdexcept>

// CUDA headers
#include <nvml.h>

namespace internal {

  [[noreturn]] inline void __abort_on_nvml_error(const char* file,
                                                 int line,
                                                 const char* cmd,
                                                 const char* error,
                                                 const char* message,
                                                 const char* description = nullptr) {
    std::ostringstream out;
    out << "\n";
    out << file << ", line " << line << ":\n";
    out << "NVML_CHECK(" << cmd << ");\n";
    out << error << ": " << message << "\n";
    if (description)
      out << description << "\n";

    throw std::runtime_error(out.str());
  }

  inline constexpr const char* __nvml_error(nvmlReturn_t result) {
    // List of error codes from CUDA version 12.9.1 and CUDA NVML headers version 12.9.79
    switch (result) {
      case NVML_SUCCESS:
        return "NVML_SUCCESS";
        break;
      case NVML_ERROR_UNINITIALIZED:
        return "NVML_ERROR_UNINITIALIZED";
        break;
      case NVML_ERROR_INVALID_ARGUMENT:
        return "NVML_ERROR_INVALID_ARGUMENT";
        break;
      case NVML_ERROR_NOT_SUPPORTED:
        return "NVML_ERROR_NOT_SUPPORTED";
        break;
      case NVML_ERROR_NO_PERMISSION:
        return "NVML_ERROR_NO_PERMISSION";
        break;
      case NVML_ERROR_ALREADY_INITIALIZED:
        return "NVML_ERROR_ALREADY_INITIALIZED";
        break;
      case NVML_ERROR_NOT_FOUND:
        return "NVML_ERROR_NOT_FOUND";
        break;
      case NVML_ERROR_INSUFFICIENT_SIZE:
        return "NVML_ERROR_INSUFFICIENT_SIZE";
        break;
      case NVML_ERROR_INSUFFICIENT_POWER:
        return "NVML_ERROR_INSUFFICIENT_POWER";
        break;
      case NVML_ERROR_DRIVER_NOT_LOADED:
        return "NVML_ERROR_DRIVER_NOT_LOADED";
        break;
      case NVML_ERROR_TIMEOUT:
        return "NVML_ERROR_TIMEOUT";
        break;
      case NVML_ERROR_IRQ_ISSUE:
        return "NVML_ERROR_IRQ_ISSUE";
        break;
      case NVML_ERROR_LIBRARY_NOT_FOUND:
        return "NVML_ERROR_LIBRARY_NOT_FOUND";
        break;
      case NVML_ERROR_FUNCTION_NOT_FOUND:
        return "NVML_ERROR_FUNCTION_NOT_FOUND";
        break;
      case NVML_ERROR_CORRUPTED_INFOROM:
        return "NVML_ERROR_CORRUPTED_INFOROM";
        break;
      case NVML_ERROR_GPU_IS_LOST:
        return "NVML_ERROR_GPU_IS_LOST";
        break;
      case NVML_ERROR_RESET_REQUIRED:
        return "NVML_ERROR_RESET_REQUIRED";
        break;
      case NVML_ERROR_OPERATING_SYSTEM:
        return "NVML_ERROR_OPERATING_SYSTEM";
        break;
      case NVML_ERROR_LIB_RM_VERSION_MISMATCH:
        return "NVML_ERROR_LIB_RM_VERSION_MISMATCH";
        break;
      case NVML_ERROR_IN_USE:
        return "NVML_ERROR_IN_USE";
        break;
      case NVML_ERROR_MEMORY:
        return "NVML_ERROR_MEMORY";
        break;
      case NVML_ERROR_NO_DATA:
        return "NVML_ERROR_NO_DATA";
        break;
      case NVML_ERROR_VGPU_ECC_NOT_SUPPORTED:
        return "NVML_ERROR_VGPU_ECC_NOT_SUPPORTED";
        break;
      case NVML_ERROR_INSUFFICIENT_RESOURCES:
        return "NVML_ERROR_INSUFFICIENT_RESOURCES";
        break;
      case NVML_ERROR_FREQ_NOT_SUPPORTED:
        return "NVML_ERROR_FREQ_NOT_SUPPORTED";
        break;
      case NVML_ERROR_ARGUMENT_VERSION_MISMATCH:
        return "NVML_ERROR_ARGUMENT_VERSION_MISMATCH";
        break;
      case NVML_ERROR_DEPRECATED:
        return "NVML_ERROR_DEPRECATED";
        break;
      case NVML_ERROR_NOT_READY:
        return "NVML_ERROR_NOT_READY";
        break;
      case NVML_ERROR_GPU_NOT_FOUND:
        return "NVML_ERROR_GPU_NOT_FOUND";
        break;
      case NVML_ERROR_INVALID_STATE:
        return "NVML_ERROR_INVALID_STATE";
        break;
      case NVML_ERROR_UNKNOWN:
        return "NVML_ERROR_UNKNOWN";
        break;
      default:
        return "NVML_UNSUPPORTED_ERROR";
    }
  }

  inline void __nvml_check(
      const char* file, int line, const char* cmd, nvmlReturn_t result, const char* description = nullptr) {
    if (result == NVML_SUCCESS)
      return;

    const char* error = __nvml_error(result);
    const char* message = nvmlErrorString(result);
    __abort_on_nvml_error(file, line, cmd, error, message, description);
  }

}  // namespace internal

#define NVML_CHECK(ARG, ...) (::internal::__nvml_check(__FILE__, __LINE__, #ARG, (ARG), ##__VA_ARGS__))
