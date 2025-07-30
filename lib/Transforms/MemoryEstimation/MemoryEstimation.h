#ifndef LIB_TRANSFORMS_MEMORYESTIMATION_MEMORYESTIMATION_H_
#define LIB_TRANSFORMS_MEMORYESTIMATION_MEMORYESTIMATION_H_

#include <map>
#include <string>

#include "llvm/include/llvm/ADT/DenseMap.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"     // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/MemoryEstimation/MemoryEstimation.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/MemoryEstimation/MemoryEstimation.h.inc"
// Forward declaration for clean separation
struct OperationLivenessInfo;
class UnifiedLivenessResults;

// Memory usage information for a single operation
struct MemoryUsage {
  uint64_t totalBytes = 0;
  uint64_t ciphertextBytes = 0;
  uint64_t keyBytes = 0;
  unsigned liveCiphertexts = 0;
  unsigned liveKeys = 0;
  unsigned totalTowers = 0;
};

// Results from memory estimation analysis
struct MemoryEstimationResults {
  llvm::DenseMap<Operation *, MemoryUsage> operationMemoryUsage;

  // Summary statistics
  uint64_t peakMemoryBytes = 0;
  uint64_t averageMemoryBytes = 0;
  uint64_t totalCiphertextMemory = 0;
  uint64_t totalKeyMemory = 0;

  // Parameters used for calculation
  unsigned ringDimension = 4096;
  unsigned dnumParameter = 2;
  bool showMatrix = false;

  // Memory distribution (bucket -> count)
  std::map<uint64_t, unsigned> memoryDistribution;

  // Report generation methods
  void generateTextReport(llvm::raw_ostream &os) const;
  void generateJSONReport(llvm::raw_ostream &os) const;

 private:
  std::string formatBytes(uint64_t bytes) const;
};

// Pass creation function
std::unique_ptr<Pass> createMemoryEstimationPass();

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_MEMORYESTIMATION_MEMORYESTIMATION_H_
