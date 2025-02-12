#include "lib/Transforms/MergeEvalKeys/MergeEvalKeys.h"

#include "lib/Analysis/RotationKeyLivenessAnalysis/RotationKeyLivenessAnalysis.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_MERGEEVALKEYS
#include "lib/Transforms/MergeEvalKeys/MergeEvalKeys.h.inc"

struct MergeEvalKeys : impl::MergeEvalKeysBase<MergeEvalKeys> {
  void runOnOperation() override {
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<lwe::KeyStateAnalysis>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      return signalPassFailure();
    }

    // Track ops to delete to avoid iterator invalidation
    SmallVector<Operation *> opsToDelete;

    auto walkResult =
        getOperation()->walk([&](openfhe::DeserializeKeyOp deserOp) {
          const auto *lattice =
              solver.lookupState<lwe::KeyStateLattice>(deserOp.getResult());
          if (!lattice || !lattice->getValue().isInitialized()) {
            return WalkResult::advance();
          }

          int64_t keyIndex = lattice->getValue().getIndex();
          PairOptimization optimization =
              findOptimizablePair(deserOp, keyIndex, solver);

          if (!optimization.isValid()) {
            return WalkResult::advance();
          }

          // Replace uses of the second deserialize with the first one
          deserOp.getResult().replaceAllUsesWith(
              optimization.firstDeser->getResult(0));

          // Track the second deserialize and first clear for deletion
          opsToDelete.push_back(deserOp);
          opsToDelete.push_back(optimization.firstClear);

          return WalkResult::advance();
        });

    if (walkResult.wasInterrupted()) {
      return signalPassFailure();
    }

    // Delete marked ops
    for (Operation *op : opsToDelete) {
      op->erase();
    }
  }

 private:
  // TODO
  // - Operation distances currently are decided by the op number. What is more
  // accurate
  //   is the latency difference. We can define cost model based on depth and
  //   ring dimension and use that to count op latency.
  // - The distance threshold for merging or leaving in separate pairs should
  // be decided carefully.
  //   Long distance means the memory will be used more and shorter distance
  //   means there will be more deserialize op.
  static constexpr int maxMergeDistance = 10;

  // Track pairs of ops we want to optimize
  struct PairOptimization {
    Operation *firstDeser = nullptr;  // First deserialize to keep
    Operation *firstClear = nullptr;  // First clear to remove

    bool isValid() const { return firstDeser && firstClear; }
  };

  PairOptimization findOptimizablePair(openfhe::DeserializeKeyOp deserOp,
                                       int64_t keyIndex,
                                       DataFlowSolver &solver) {
    PairOptimization result;
    // Block *block = deserOp->getBlock();
    Operation *currentOp = deserOp;
    int distance = 0;

    // Track the clear op associated with our deserialize
    Operation *secondClear = nullptr;
    for (Operation *user : deserOp.getResult().getUsers()) {
      if (auto clearOp = dyn_cast<openfhe::ClearKeyOp>(user)) {
        secondClear = clearOp;
        break;
      }
    }
    if (!secondClear) return result;

    // Walk backwards looking for matching deserialize/clear pair
    while ((currentOp = currentOp->getPrevNode()) &&
           distance < maxMergeDistance) {
      distance++;

      if (auto existingDeser = dyn_cast<openfhe::DeserializeKeyOp>(currentOp)) {
        const auto *lattice =
            solver.lookupState<lwe::KeyStateLattice>(existingDeser.getResult());
        if (lattice && lattice->getValue().isInitialized() &&
            lattice->getValue().getIndex() == keyIndex) {
          // Find its associated clear op
          Operation *firstClear = nullptr;
          for (Operation *user : existingDeser.getResult().getUsers()) {
            if (auto clearOp = dyn_cast<openfhe::ClearKeyOp>(user)) {
              firstClear = clearOp;
              break;
            }
          }

          if (!firstClear) continue;

          // Verify firstClear comes before our second deserialize
          if (firstClear->isBeforeInBlock(deserOp)) {
            result.firstDeser = existingDeser;
            result.firstClear = firstClear;
            return result;
          }
        }
      }
    }
    return result;
  }
};
}  // namespace heir
}  // namespace mlir
