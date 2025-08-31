// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

int main(int argc, char* argv[]) {
  // run the Catch tests
  return Catch::Session().run(argc, argv);
}
