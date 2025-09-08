/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <iomanip>
#include <iostream>
#include <string>

// CUDA headers
#include <cuda_runtime.h>
#include <nvml.h>

// local headers
#include "cuda_check.h"
#include "nvml_check.h"

namespace internal {

  // NVIDIA system driver version, e.g. 575.57.08
  inline std::string __nvidia_driver_version() {
    std::string version(NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE, '\0');
    NVML_CHECK(nvmlInitWithFlags(NVML_INIT_FLAG_NO_GPUS | NVML_INIT_FLAG_NO_ATTACH));
    NVML_CHECK(nvmlSystemGetDriverVersion(version.data(), version.size()));
    NVML_CHECK(nvmlShutdown());
    std::string::size_type null = version.find_first_of('\0');
    if (null != std::string::npos) {
      version.resize(null);
    }
    return version;
  }

  // CUDA driver version, e.g. 12.9
  inline std::string __cuda_driver_version() {
    int version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&version));
    return std::to_string(version / 1000) + '.' + std::to_string(version % 1000 / 10);
  }

  // CUDA runtime version, e.g. 12.9
  inline std::string __cuda_runtime_version() {
    int version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&version));
    return std::to_string(version / 1000) + '.' + std::to_string(version % 1000 / 10);
  }

}  // namespace internal

// NVIDIA system driver version, e.g. 575.57.08
inline std::string const& nvidia_driver_version() {
  static std::string version = internal::__nvidia_driver_version();
  return version;
}

// CUDA driver version, e.g. 12.9
inline std::string const& cuda_driver_version() {
  static std::string version = internal::__cuda_driver_version();
  return version;
}

// CUDA runtime version, e.g. 12.9
inline std::string const& cuda_runtime_version() {
  static std::string version = internal::__cuda_runtime_version();
  return version;
}

// CUDA version
inline std::string const& cuda_version() {
  static std::string version = std::string("NVIDIA driver ") + nvidia_driver_version() + ", CUDA driver " +
                               cuda_driver_version() + ", CUDA runtime " + cuda_runtime_version();
  return version;
}
