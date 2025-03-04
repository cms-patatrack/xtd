
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
  XTD_DEVICE_FUNCTION
  inline constexpr float acosh(T arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::acosh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::acosh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::acosh(arg);
#else
    // standard C++ code
    return std::acosh(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr float acoshf(float arg) {
    return acosh(arg);
  }

  XTD_DEVICE_FUNCTION
  inline constexpr long double acoshl(long double arg) {
    return acosh(arg);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  XTD_DEVICE_FUNCTION
  inline constexpr double acosh(T arg) {
    return acosh(static_cast<double>(arg));
  }

}  // namespace xtd
