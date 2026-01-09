/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <iomanip>
#include <iostream>
#include <string>

// ROCm headers
#include <hip_runtime.h>
#include <rocm_smi/rocm_smi.h>

// local headers
#include "hip_check.h"
#include "rsmi_check.h"

namespace test::hip {

  namespace internal {

    // AMDGPU driver version, e.g. 6.12.12
    inline std::string __amdgpu_driver_version() {
      std::string version(256, '\0');
      RSMI_CHECK(rsmi_init(0x00));
      RSMI_CHECK(rsmi_version_str_get(RSMI_SW_COMP_DRIVER, version.data(), version.size()));
      RSMI_CHECK(rsmi_shut_down());
      // trim trailing null characters
      std::string::size_type null = version.find_first_of('\0');
      if (null != std::string::npos) {
        version.resize(null);
      }
      return version;
    }

    // ROCm driver version, e.g. 6.3.42134
    inline std::string __hip_driver_version() {
      int version = 0;
      HIP_CHECK(hipDriverGetVersion(&version));
      // decode 60342134 as 6.3.42134
      return std::to_string(version / 10000000) + '.' + std::to_string(version / 100000 % 100) + '.' +
             std::to_string(version % 100000);
    }

    // ROCm runtime version, e.g. 6.3.42134
    inline std::string __hip_runtime_version() {
      int version = 0;
      HIP_CHECK(hipRuntimeGetVersion(&version));
      // decode 60342134 as 6.3.42134
      return std::to_string(version / 10000000) + '.' + std::to_string(version / 100000 % 100) + '.' +
             std::to_string(version % 100000);
    }

  }  // namespace internal

  // AMDGPU driver version, e.g. 6.12.12
  inline std::string const& amdgpu_driver_version() {
    static std::string version = internal::__amdgpu_driver_version();
    return version;
  }

  // ROCm driver version, e.g. 6.3.42134
  inline std::string const& hip_driver_version() {
    static std::string version = internal::__hip_driver_version();
    return version;
  }

  // ROCm runtime version, e.g. 6.3.42134
  inline std::string const& hip_runtime_version() {
    static std::string version = internal::__hip_runtime_version();
    return version;
  }

  // ROCm version
  inline std::string const& version() {
    static std::string version = std::string("AMDGPU driver ") + amdgpu_driver_version() + ", ROCm driver " +
                                 hip_driver_version() + ", ROCm runtime " + hip_runtime_version();
    return version;
  }

}  // namespace test::hip
