#ifndef LIB_ANALYSIS_ROTATIONKEYANALYSIS_ROTATIONANALYSIS_H_
#define LIB_ANALYSIS_ROTATIONKEYANALYSIS_ROTATIONANALYSIS_H_

#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Liveness.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"                // from @llvm-project

#define DEBUG_TYPE "rotation-key-liveness"

namespace mlir {
namespace heir {

class RotationKeyInterval {
 public:
  // First operation that uses this rotation index
  Operation* firstUse = nullptr;
  // Last operation that uses this rotation index
  Operation* lastUse = nullptr;
  // Operations that use this rotation index
  SmallVector<tensor_ext::RotateOp, 4> uses;
};

class RotationKeyLivenessAnalysis {
 public:
  RotationKeyLivenessAnalysis(Operation* op) {
    // Use constant propagation analysis to determine constant shift values
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    if (failed(solver.initializeAndRun(op))) {
      op->emitOpError() << "Failed to run constant propagation analysis";
      return;
    }

    // Collect all rotation ops and determine their constant indices
    SmallVector<Operation*> orderedOps;
    op->walk<WalkOrder::PreOrder>([&](Operation* currentOp) {
      orderedOps.push_back(currentOp);

      if (auto rotateOp = dyn_cast<tensor_ext::RotateOp>(currentOp)) {
        // Get constant value analysis result for shift operand
        Value shift = rotateOp.getShift();
        const auto* lattice =
            solver.lookupState<dataflow::Lattice<dataflow::ConstantValue>>(
                shift);

        if (lattice && !lattice->getValue().isUninitialized() &&
            lattice->getValue().getConstantValue()) {
          // If we can determine the constant shift value
          auto shiftAttr =
              dyn_cast<IntegerAttr>(lattice->getValue().getConstantValue());
          if (shiftAttr) {
            int64_t index = shiftAttr.getInt();

            auto& interval = rotationKeyIntervals[index];
            interval.uses.push_back(rotateOp);

            // Update first/last use
            if (!interval.firstUse) {
              interval.firstUse = currentOp;
            }
            interval.lastUse = currentOp;
          }
        } else {
          // Track rotation ops where we can't determine the shift statically
          dynamicShiftOps.push_back(rotateOp);
        }
      }
    });
  }

  // Get the lifetime interval for a rotation index
  const RotationKeyInterval* getInterval(int64_t index) const {
    auto it = rotationKeyIntervals.find(index);
    if (it != rotationKeyIntervals.end()) {
      return &it->second;
    }
    return nullptr;
  }

  // Get rotation ops with non-constant shifts
  ArrayRef<tensor_ext::RotateOp> getDynamicShiftOps() const {
    return dynamicShiftOps;
  }

  // Check if a rotation key is needed at this operation
  bool isKeyNeededAt(int64_t index, Operation* op) const {
    auto interval = getInterval(index);
    if (!interval) return false;

    // Key is needed if op is between first and last use
    return op->isBeforeInBlock(interval->lastUse) &&
           interval->firstUse->isBeforeInBlock(op);
  }

  // Returns all rotation indices used in the program
  SmallVector<int64_t> getAllRotationIndices() const {
    SmallVector<int64_t> indices;
    for (const auto& [index, _] : rotationKeyIntervals) {
      indices.push_back(index);
    }
    return indices;
  }

  void dump() const;

 private:
  // Map from rotation indices to their lifetime intervals
  DenseMap<int64_t, RotationKeyInterval> rotationKeyIntervals;

  // Rotation ops with non-constant shifts
  SmallVector<tensor_ext::RotateOp, 4> dynamicShiftOps;
};
}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_ROTATIONKEYANALYSIS_ROTATIONANALYSIS_H_
