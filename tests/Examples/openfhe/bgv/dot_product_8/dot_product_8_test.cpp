#include <cstdint>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "src/pke/include/openfhe.h"  // from @openfhe
#include "tests/Examples/openfhe/KeyMemRT.hpp"
#include "tests/Examples/openfhe/ResourceMonitor.hpp"
KeyMemRT keymem_rt;

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/bgv/dot_product_8/dot_product_8_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

TEST(DotProduct8Test, RunTest) {
  auto mode = KeyMemMode::IGNORE;
  keymem_rt.setKeyMemMode(mode);
  ResourceMonitor monitor;
  monitor.start();

  auto cryptoContext = dot_product__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      dot_product__configure_crypto_context(cryptoContext, secretKey);

  std::vector<int16_t> arg0 = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int16_t> arg1 = {2, 3, 4, 5, 6, 7, 8, 9};
  int64_t expected = 240;

  auto arg0Encrypted =
      dot_product__encrypt__arg0(cryptoContext, arg0, publicKey);
  auto arg1Encrypted =
      dot_product__encrypt__arg1(cryptoContext, arg1, publicKey);
  auto outputEncrypted =
      dot_product(cryptoContext, arg0Encrypted, arg1Encrypted);
  monitor.stop();
  std::string filename =
      "./resource_usage_dot_product_8_" + getModeString(mode) + ".csv";
  monitor.save_to_file(filename);
  auto actual =
      dot_product__decrypt__result0(cryptoContext, outputEncrypted, secretKey);

  EXPECT_EQ(expected, actual);
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
