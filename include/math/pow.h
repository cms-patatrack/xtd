
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr T pow(T base, T exp) {
#if defined(XTD_TARGET_CUDA)
  // CUDA device code
  return ::pow(base, exp);
#elif defined(XTD_TARGET_HIP)
  // HIP/ROCm device code
  return ::pow(base, exp);
#elif defined(XTD_TARGET_SYCL)
  // SYCL device code
  return sycl::pow(base, exp);
#else
  // standard C++ code
  return std::pow(base, exp);
#endif
}

XTD_DEVICE_FUNCTION inline constexpr float powf(float base, float exp) {
  return powf(base, exp);
}

XTD_DEVICE_FUNCTION inline constexpr long double powl(long double base, long double exp) {
  return powl(base, exp);
}

} // namespace xtd
