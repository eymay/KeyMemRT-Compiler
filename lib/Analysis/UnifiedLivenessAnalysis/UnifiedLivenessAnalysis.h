#ifndef LIB_ANALYSIS_UNIFIEDLIVENESSANALYSIS_UNIFIEDLIVENESSANALYSIS_H_
#define LIB_ANALYSIS_UNIFIEDLIVENESSANALYSIS_UNIFIEDLIVENESSANALYSIS_H_

#include <set>
#include <vector>

#include "llvm/include/llvm/ADT/DenseMap.h"         // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project

namespace mlir {
namespace heir {

// Result structure for easy consumption by other passes
struct OperationLivenessInfo {
  Operation *op;

  // Ciphertext information
  unsigned liveCiphertextCount = 0;
  unsigned totalCiphertextTowers = 0;
  std::vector<unsigned> ciphertextTowerCounts;  // Individual tower counts

  // Key information
  unsigned liveKeyCount = 0;
  unsigned totalKeyTowers = 0;
  std::vector<unsigned> keyTowerCounts;     // Individual tower counts
  std::vector<int64_t> keyRotationIndices;  // Rotation indices for keys
  std::vector<unsigned> keyDepths;          // Key depths for tower calculation

  // Summary
  unsigned maxTowerNumber = 0;
  unsigned totalTowers = 0;

  OperationLivenessInfo(Operation *operation) : op(operation) {}
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

// Simple liveness state - following the pattern of RotationDepthAnalysis
class LivenessState {
 public:
  LivenessState() = default;

  // Simple tower tracking - just like depthMap in RotationDepthAnalysis
  llvm::DenseMap<Value, unsigned> ciphertextTowers;
  llvm::DenseMap<Value, unsigned> keyTowers;
  llvm::DenseMap<Value, bool> isKeyMap;  // Track which values are keys

  // Additional key information for analysis results
  llvm::DenseMap<Value, int64_t> keyRotationIndices;
  llvm::DenseMap<Value, unsigned> keyDepths;
  llvm::DenseMap<int64_t, unsigned> globalKeyTowers;
  // Tower calculations
  unsigned getTotalTowers() const;
  unsigned getMaxTowerNumber() const;
  unsigned getCiphertextCount() const;
  unsigned getKeyCount() const;

  // State operations for dataflow - simple like KeyState
  bool operator==(const LivenessState &other) const;
  static LivenessState join(const LivenessState &lhs, const LivenessState &rhs);

  // Convert to structured result for consumption
  OperationLivenessInfo toOperationInfo(Operation *op) const;

  void print(raw_ostream &os) const;
};

// Dataflow lattice element for liveness state
class LivenessLattice : public dataflow::Lattice<LivenessState> {
 public:
  using Lattice::Lattice;
};

// Main dataflow analysis for unified liveness
class UnifiedLivenessAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<LivenessLattice> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const LivenessLattice *> operands,
                               ArrayRef<LivenessLattice *> results) override;

  void setToEntryState(LivenessLattice *lattice) override;

 private:
  // Helper methods for tower calculations
  unsigned getCiphertextTowerCount(Value value) const;
  unsigned getKeyTowerCount(Value value, unsigned multDepth = 10) const;

  // Type checking helpers
  bool isCiphertextType(Type type) const;
  bool isKeyType(Type type) const;

  // Key-specific helpers
  unsigned getKeyDepthFromDeserialize(Operation *deserializeOp) const;
  int64_t getKeyRotationIndex(Operation *deserializeOp) const;
  unsigned calculateKeyQTowers(unsigned multDepth, unsigned level) const;

  // Configuration
  static constexpr unsigned FIXED_P_TOWERS = 4;
};

// Helper function to run analysis and get results
UnifiedLivenessResults runUnifiedLivenessAnalysis(Operation *op);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_UNIFIEDLIVENESSANALYSIS_UNIFIEDLIVENESSANALYSIS_H_
