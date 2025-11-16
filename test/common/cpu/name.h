/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <fstream>
#include <string>

namespace internal {

  /* Parse /proc/cpuinfo and look for athe CPU model name.
     */
  inline std::string cpu_get_name() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo)
      return {};

    std::string line;
    while (std::getline(cpuinfo, line)) {
      if (line.starts_with("model name")) {
        auto pos = line.find(':');
        if (pos == std::string::npos)
          return {};

        auto model = line.substr(pos + 1);

        // trim leading and trailing spaces and tabs
        auto first = model.find_first_not_of(" \t");
        if (first == std::string::npos)
          return {};  // line had only whitespace

        auto last = model.find_last_not_of(" \t");
        model = model.substr(first, last - first + 1);

        return model;
      }
    }

    return {};
  }

}  // namespace internal

inline std::string cpu_get_name() {
  static std::string name = internal::cpu_get_name();

  return name;
}
