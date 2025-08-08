#include "lib/Transforms/MergeEvalKeys/MergeEvalKeys.h"

#include "lib/Analysis/RotationKeyLivenessAnalysis/RotationKeyLivenessAnalysis.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project

#define DEBUG_TYPE "merge-eval-keys"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_MERGEEVALKEYS
#include "lib/Transforms/MergeEvalKeys/MergeEvalKeys.h.inc"

// Structure to hold optimization results with distance stats
struct PairOptimizationWithStats {
  Operation *firstDeser = nullptr;  // First deserialize to keep
  Operation *firstClear = nullptr;  // First clear to remove
  int realDistance = 0;             // Distance excluding key ops
  int totalDistance = 0;            // Total operation count

  bool isValid() const { return firstDeser && firstClear; }
};

struct MergeEvalKeys : impl::MergeEvalKeysBase<MergeEvalKeys> {
  void runOnOperation() override {
    bool madeChanges = true;
    unsigned iterationCount = 0;

    // Keep iterating until no more pairs can be merged
    while (madeChanges) {
      madeChanges = false;
      iterationCount++;

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
            PairOptimizationWithStats optimization =
                findOptimizablePairWithDistance(deserOp, keyIndex, solver);

            if (!optimization.isValid()) {
              return WalkResult::advance();
            }
            auto firstDeserOp =
                cast<openfhe::DeserializeKeyOp>(optimization.firstDeser);
            if (deserOp->hasAttr("key_depth") &&
                firstDeserOp->hasAttr("key_depth")) {
              auto firstLevel =
                  firstDeserOp->getAttrOfType<IntegerAttr>("key_depth")
                      .getInt();
              auto secondLevel =
                  deserOp->getAttrOfType<IntegerAttr>("key_depth").getInt();
              if (firstLevel != secondLevel) {
                return WalkResult::advance();  // Skip merging different levels
              }
            } else if (!firstDeserOp->hasAttr("key_depth") &&
                       deserOp->hasAttr("key_depth")) {
              firstDeserOp->setAttr("key_depth", deserOp->getAttr("key_depth"));
            }
            // Replace uses of the second deserialize with the first one
            deserOp.getResult().replaceAllUsesWith(
                optimization.firstDeser->getResult(0));

            // Track the second deserialize and first clear for deletion
            opsToDelete.push_back(deserOp);
            opsToDelete.push_back(optimization.firstClear);

            // Mark that we made changes
            madeChanges = true;

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

    // Print statistics
    LLVM_DEBUG(llvm::dbgs() << "MergeEvalKeys: Completed in " << iterationCount
                            << " iterations\n";);
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

  // Helper to check if an operation is a key management operation
  bool isKeyManagementOp(Operation *op) {
    return isa<openfhe::DeserializeKeyOp, openfhe::ClearKeyOp>(op);
  }

  PairOptimization findOptimizablePair(openfhe::DeserializeKeyOp deserOp,
                                       int64_t keyIndex,
                                       DataFlowSolver &solver) {
    PairOptimization result;
    Operation *currentOp = deserOp;
    int distance = 0;
    int realDistance = 0;  // Distance excluding key management ops

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
           realDistance < maxMergeDistance) {
      // Count distance - skip key management operations
      if (!isKeyManagementOp(currentOp)) {
        realDistance++;
      }

      distance++;  // Total distance for debugging

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

            // Debug output
            llvm::outs() << "Found pair for key index " << keyIndex
                         << " with real distance " << realDistance
                         << " (total ops: " << distance << ")\n";
            return result;
          }
        }
      }
    }
    return result;
  }

  // Alternative version that counts operations between clear and deserialize
  PairOptimizationWithStats findOptimizablePairWithDistance(
      openfhe::DeserializeKeyOp deserOp, int64_t keyIndex,
      DataFlowSolver &solver) {
    PairOptimizationWithStats result;
    Operation *currentOp = deserOp;

    // Track the clear op associated with our deserialize
    Operation *secondClear = nullptr;
    for (Operation *user : deserOp.getResult().getUsers()) {
      if (auto clearOp = dyn_cast<openfhe::ClearKeyOp>(user)) {
        secondClear = clearOp;
        break;
      }
    }
    if (!secondClear) return result;

    // Find matching deserialize/clear pair
    while ((currentOp = currentOp->getPrevNode())) {
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
            // Count operations between firstClear and deserOp
            int realDistance = 0;
            int totalOps = 0;

            Operation *walkOp = firstClear->getNextNode();
            while (walkOp && walkOp != deserOp) {
              totalOps++;
              if (!isKeyManagementOp(walkOp)) {
                realDistance++;
              }
              walkOp = walkOp->getNextNode();
            }

            // Check if within distance threshold
            if (realDistance <= maxMergeDistance) {
              result.firstDeser = existingDeser;
              result.firstClear = firstClear;
              result.realDistance = realDistance;
              result.totalDistance = totalOps;

              LLVM_DEBUG(llvm::dbgs()
                             << "Found optimal pair for key index " << keyIndex
                             << " with real distance " << realDistance
                             << " (total ops: " << totalOps << ")\n";);
              return result;
            }
          }
        }
      }
    }
    return result;
  }
};
}  // namespace heir
}  // namespace mlir
