#ifndef LIB_ANALYSIS_OPTIMIZE_ROTATIONBASEANALYSIS_H
#define LIB_ANALYSIS_OPTIMIZE_ROTATIONBASEANALYSIS_H

#include <map>
#include <vector>

#include "llvm/include/llvm/ADT/DenseMap.h"
#include "llvm/include/llvm/ADT/STLExtras.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/LogicalResult.h"

namespace mlir {
namespace heir {

class RotationBaseAnalysis {
 public:
  RotationBaseAnalysis(Operation *op, int baseSetSize = 2)
      : opToRunOn(op), baseSetSize(baseSetSize) {}
  ~RotationBaseAnalysis() = default;

  // Solves the optimization problem to find the optimal base set
  LogicalResult solve();

  // Returns the optimal base set after solving
  const std::vector<int64_t> &getOptimalBaseSet() const {
    return optimalBaseSet;
  }

  // Returns true if the index is in the optimal base set
  bool isInBaseSet(int64_t index) const {
    return std::find(optimalBaseSet.begin(), optimalBaseSet.end(), index) !=
           optimalBaseSet.end();
  }

  // Returns the composition of a rotation index in terms of base rotations
  // The returned map maps from base rotation index to count of operations
  std::map<int64_t, int64_t> getComposition(int64_t index) const;

  // Returns the total number of operations needed for a rotation
  int64_t getTotalOperations(int64_t index) const;

  // Make allRotationIndices public so the pass can iterate through them
  std::vector<int64_t> allRotationIndices;

 private:
  // Collects all rotation indices from the IR
  void collectRotationIndices();

  Operation *opToRunOn;
  int baseSetSize;

  // The optimal base set after solving
  std::vector<int64_t> optimalBaseSet;

  // Maps each rotation index to its composition in terms of base rotations
  // Map structure: rotation index -> (base index -> count)
  llvm::DenseMap<int64_t, std::map<int64_t, int64_t>> compositionMappings;
};

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_OPTIMIZE_ROTATIONBASEANALYSIS_H
