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

// ROCm headers
#include <hip/hip_version.h>
#include <rocm_smi/rocm_smi.h>

namespace internal {

  [[noreturn]] inline void __abort_on_rsmi_error(const char* file,
                                                 int line,
                                                 const char* cmd,
                                                 const char* error,
                                                 const char* message,
                                                 const char* description = nullptr) {
    std::ostringstream out;
    out << "\n";
    out << file << ", line " << line << ":\n";
    out << "RSMI_CHECK(" << cmd << ");\n";
    out << error << ": " << message << "\n";
    if (description)
      out << description << "\n";

    throw std::runtime_error(out.str());
  }

  inline constexpr const char* __rsmi_error(rsmi_status_t result) {
    // List of error codes from ROCm version 6.4.1 and ROCm SMI library version 7.5.0
    switch (result) {
      case RSMI_STATUS_SUCCESS:
        return "RSMI_STATUS_SUCCESS";
        break;
      case RSMI_STATUS_INVALID_ARGS:
        return "RSMI_STATUS_INVALID_ARGS";
        break;
      case RSMI_STATUS_NOT_SUPPORTED:
        return "RSMI_STATUS_NOT_SUPPORTED";
        break;
      case RSMI_STATUS_FILE_ERROR:
        return "RSMI_STATUS_FILE_ERROR";
        break;
      case RSMI_STATUS_PERMISSION:
        return "RSMI_STATUS_PERMISSION";
        break;
      case RSMI_STATUS_OUT_OF_RESOURCES:
        return "RSMI_STATUS_OUT_OF_RESOURCES";
        break;
      case RSMI_STATUS_INTERNAL_EXCEPTION:
        return "RSMI_STATUS_INTERNAL_EXCEPTION";
        break;
      case RSMI_STATUS_INPUT_OUT_OF_BOUNDS:
        return "RSMI_STATUS_INPUT_OUT_OF_BOUNDS";
        break;
      case RSMI_STATUS_INIT_ERROR:
        return "RSMI_STATUS_INIT_ERROR";
        break;
      case RSMI_STATUS_NOT_YET_IMPLEMENTED:
        return "RSMI_STATUS_NOT_YET_IMPLEMENTED";
        break;
      case RSMI_STATUS_NOT_FOUND:
        return "RSMI_STATUS_NOT_FOUND";
        break;
      case RSMI_STATUS_INSUFFICIENT_SIZE:
        return "RSMI_STATUS_INSUFFICIENT_SIZE";
        break;
      case RSMI_STATUS_INTERRUPT:
        return "RSMI_STATUS_INTERRUPT";
        break;
      case RSMI_STATUS_UNEXPECTED_SIZE:
        return "RSMI_STATUS_UNEXPECTED_SIZE";
        break;
      case RSMI_STATUS_NO_DATA:
        return "RSMI_STATUS_NO_DATA";
        break;
      case RSMI_STATUS_UNEXPECTED_DATA:
        return "RSMI_STATUS_UNEXPECTED_DATA";
        break;
      case RSMI_STATUS_BUSY:
        return "RSMI_STATUS_BUSY";
        break;
      case RSMI_STATUS_REFCOUNT_OVERFLOW:
        return "RSMI_STATUS_REFCOUNT_OVERFLOW";
        break;
#if HIP_VERSION >= 50600000
      // ROCm 5.6.0
      case RSMI_STATUS_SETTING_UNAVAILABLE:
        return "RSMI_STATUS_SETTING_UNAVAILABLE";
        break;
      case RSMI_STATUS_AMDGPU_RESTART_ERR:
        return "RSMI_STATUS_AMDGPU_RESTART_ERR";
        break;
#endif
#if HIP_VERSION >= 60400000
      // ROCm 6.4.0
      case RSMI_STATUS_DRM_ERROR:
        return "RSMI_STATUS_DRM_ERROR";
        break;
#endif
#if HIP_VERSION >= 60443483
      // ROCm 6.4.1
      case RSMI_STATUS_FAIL_LOAD_MODULE:
        return "RSMI_STATUS_FAIL_LOAD_MODULE";
        break;
      case RSMI_STATUS_FAIL_LOAD_SYMBOL:
        return "RSMI_STATUS_FAIL_LOAD_SYMBOL";
        break;
#endif
      case RSMI_STATUS_UNKNOWN_ERROR:
        return "RSMI_STATUS_UNKNOWN_ERROR";
        break;
      default:
        return "RSMI_STATUS_UNKNOWN_ERROR";
    }
  }

  inline void __rsmi_check(
      const char* file, int line, const char* cmd, rsmi_status_t result, const char* description = nullptr) {
    if (result == RSMI_STATUS_SUCCESS)
      return;
    const char* error = __rsmi_error(result);
    const char* message;
    rsmi_status_string(result, &message);
    __abort_on_rsmi_error(file, line, cmd, error, message, description);
  }

}  // namespace internal

#define RSMI_CHECK(ARG, ...) (internal::__rsmi_check(__FILE__, __LINE__, #ARG, (ARG), ##__VA_ARGS__))
