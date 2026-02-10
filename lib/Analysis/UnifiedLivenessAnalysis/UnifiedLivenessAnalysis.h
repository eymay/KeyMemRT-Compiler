#ifndef LIB_ANALYSIS_UNIFIEDLIVENESSANALYSIS_UNIFIEDLIVENESSANALYSIS_H_
#define LIB_ANALYSIS_UNIFIEDLIVENESSANALYSIS_UNIFIEDLIVENESSANALYSIS_H_

#include <vector>

#include "llvm/include/llvm/ADT/DenseMap.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "mlir/include/mlir/IR/Operation.h"

namespace mlir {
namespace heir {

// Information about liveness at a single operation
struct OperationLivenessInfo {
  Operation *op;

  // Operation identification information
  std::string resultName;  // e.g., "%ct_901", "%904"
  unsigned timestep = 0;   // Timestep number in the analysis

  // Ciphertext information
  unsigned liveCiphertextCount = 0;
  std::vector<unsigned> ciphertextTowerCounts;
  unsigned totalCiphertextTowers = 0;

  // Key information
  unsigned liveKeyCount = 0;
  std::vector<unsigned> keyTowerCounts;
  std::vector<int64_t> keyRotationIndices;  // key indices that are live
  unsigned totalKeyTowers = 0;

  // Summary information
  unsigned maxTowerNumber = 0;
  unsigned totalTowers = 0;

  explicit OperationLivenessInfo(Operation *operation) : op(operation) {}

  void print(raw_ostream &os) const;
};

// Analysis results that other passes can easily consume
class UnifiedLivenessResults {
 public:
  std::vector<OperationLivenessInfo> operationResults;

  // Helper methods
  const OperationLivenessInfo *getInfoForOperation(Operation *op) const;
  void addOperationInfo(const OperationLivenessInfo &info);

  // Summary statistics
  unsigned getMaxTowerCount() const;
  unsigned getTotalOperations() const { return operationResults.size(); }

  void print(raw_ostream &os) const;
};

// Simple liveness state - no dataflow needed
class LivenessState {
 public:
  LivenessState() = default;

  // Simple tower tracking
  llvm::DenseMap<Value, unsigned>
      ciphertextTowers;                        // ciphertext -> tower_count
  llvm::DenseMap<int64_t, unsigned> liveKeys;  // key_index -> tower_count

  // Count functions - return number of items
  unsigned getCiphertextCount() const;
  unsigned getKeyCount() const;

  // Tower functions - return total towers
  unsigned getCiphertextTowers() const;
  unsigned getKeyTowers() const;
  unsigned getTotalTowers() const;
  unsigned getMaxTowerNumber() const;

  // Convert to structured result for consumption
  OperationLivenessInfo toOperationInfo(Operation *op) const;

  void print(raw_ostream &os) const;
};

// Helper function to run analysis and get results
UnifiedLivenessResults runUnifiedLivenessAnalysis(Operation *op);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_UNIFIEDLIVENESSANALYSIS_UNIFIEDLIVENESSANALYSIS_H_
