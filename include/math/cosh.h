
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
  XTD_DEVICE_FUNCTION
  inline constexpr T cosh(T arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::cosh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::cosh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cosh(arg);
#else
    // standard C++ code
    return std::cosh(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr float coshf(float arg) {
    return cosh(arg);
  }

  XTD_DEVICE_FUNCTION
  inline constexpr long double coshl(long double arg) {
    return cosh(arg);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  XTD_DEVICE_FUNCTION
  inline constexpr double cosh(T arg) {
    return cosh(static_cast<double>(arg));
  }

}  // namespace xtd
