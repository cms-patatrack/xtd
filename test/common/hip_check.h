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

// HIP headers
#include <hip/hip_runtime.h>

namespace internal {

  [[noreturn]] inline void __abort_on_hip_error(const char *file,
                                                int line,
                                                const char *cmd,
                                                const char *error,
                                                const char *message,
                                                const char *description = nullptr) {
    std::ostringstream out;
    out << "\n";
    out << file << ", line " << line << ":\n";
    out << "HIP_CHECK(" << cmd << ");\n";
    out << error << ": " << message << "\n";
    if (description)
      out << description << "\n";

    throw std::runtime_error(out.str());
  }

  inline void __hip_check(
      const char *file, int line, const char *cmd, hipError_t result, const char *description = nullptr) {
    if (result == hipSuccess)
      return;

    const char *error = hipGetErrorName(result);
    const char *message = hipGetErrorString(result);
    __abort_on_hip_error(file, line, cmd, error, message, description);
  }

}  // namespace internal

#define HIP_CHECK(ARG, ...) (internal::__hip_check(__FILE__, __LINE__, #ARG, (ARG), ##__VA_ARGS__))
