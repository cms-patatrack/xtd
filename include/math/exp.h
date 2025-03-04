
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr T exp(T x) {
#if defined(XTD_TARGET_CUDA)
  // CUDA device code
  return ::exp(x);
#elif defined(XTD_TARGET_HIP)
  // HIP/ROCm device code
  return ::exp(x);
#elif defined(XTD_TARGET_SYCL)
  // SYCL device code
  return sycl::exp(x);
#else
  // standard C++ code
  return std::exp(x);
#endif
}

XTD_DEVICE_FUNCTION inline constexpr float expf(float x) {
  return exp(x);
}

XTD_DEVICE_FUNCTION inline constexpr long double expl(long double x) {
  return exp(x);
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr double exp(T x) {
	return exp(static_cast<double>(x));
}

} // namespace xtd
