#pragma once

// C++ standard headers
#include <iostream>
#include <sstream>
#include <stdexcept>

// HIP headers
#include <hip/hip_runtime.h>

namespace internal {

  [[noreturn]] inline void abortOnHipError(const char *file,
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

  inline void hipCheck(const char *file,
                       int line,
                       const char *cmd,
                       hipError_t result,
                       const char *description = nullptr) {
    if (result == hipSuccess)
      return;

    const char *error = hipGetErrorName(result);
    const char *message = hipGetErrorString(result);
    abortOnHipError(file, line, cmd, error, message, description);
  }

}  // namespace internal

#define HIP_CHECK(ARG, ...) (internal::hipCheck(__FILE__, __LINE__, #ARG, (ARG), ##__VA_ARGS__))
