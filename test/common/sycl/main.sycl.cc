#include <cstdlib>
#include <iostream>

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

// SYCL headers
#include <sycl/sycl.hpp>

int main(int argc, char *argv[]) {
  // check that the system has one or more SYCL devices
  int deviceCount = 0;
  for (const auto &platform : sycl::platform::get_platforms()) {
    for (const auto &device : platform.get_devices()) {
      ++deviceCount;
      if (not device.has(sycl::aspect::fp64)) {
        std::cout << "The device " << device.get_info<sycl::info::device::name>() << " on the platform "
                  << platform.get_info<sycl::info::platform::name>()
                  << " does not support double precision floating point operations, some tests will be skipped."
                  << std::endl;
      }
    }
  }

  if (deviceCount == 0) {
    std::cout << "No SYCL devises found, the test will be skipped.\n\n";
    exit(EXIT_SUCCESS);
  }

  // run the Catch tests
  return Catch::Session().run(argc, argv);
}
