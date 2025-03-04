
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
  XTD_DEVICE_FUNCTION
  inline constexpr T tan(T arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::tan(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::tan(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::tan(arg);
#else
    // standard C++ code
    return std::tan(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr float tanf(float arg) {
    return tan(arg);
  }

  XTD_DEVICE_FUNCTION
  inline constexpr long double tanl(long double arg) {
    return tan(arg);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  XTD_DEVICE_FUNCTION
  inline constexpr double tan(T arg) {
    return tan(static_cast<double>(arg));
  }

}  // namespace xtd
