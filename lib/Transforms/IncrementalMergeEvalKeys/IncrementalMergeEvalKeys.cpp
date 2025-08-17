// lib/Transforms/MergeEvalKeys/IncrementalMergeEvalKeys.cpp
#include "lib/Transforms/IncrementalMergeEvalKeys/IncrementalMergeEvalKeys.h"

#include "lib/Analysis/RotationKeyLivenessAnalysis/RotationKeyLivenessAnalysis.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "llvm/include/llvm/Support/Debug.h"
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/Support/LLVM.h"

#define DEBUG_TYPE "incremental-merge-eval-keys"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_INCREMENTALMERGEEVALKEYS
#include "lib/Transforms/IncrementalMergeEvalKeys/IncrementalMergeEvalKeys.h.inc"

// Work range tracking for incremental analysis
struct WorkRange {
  Operation *start;
  Operation *end;

  WorkRange(Operation *s, Operation *e) : start(s), end(e) {}

  bool contains(Operation *op) const {
    if (!start || !end || !op) return false;
    // Ensure all operations are in the same block
    if (start->getBlock() != end->getBlock() ||
        op->getBlock() != start->getBlock()) {
      return false;
    }
    return start->isBeforeInBlock(op) && op->isBeforeInBlock(end);
  }

  bool overlaps(const WorkRange &other) const {
    if (!start || !end || !other.start || !other.end) return false;
    // Ensure all operations are in the same block
    if (start->getBlock() != end->getBlock() ||
        other.start->getBlock() != other.end->getBlock() ||
        start->getBlock() != other.start->getBlock()) {
      return false;
    }
    return !(end->isBeforeInBlock(other.start) ||
             other.end->isBeforeInBlock(start));
  }

  // Expand this range to include another range
  void merge(const WorkRange &other) {
    if (!other.start || !other.end) return;
    if (!start || other.start->isBeforeInBlock(start)) {
      start = other.start;
    }
    if (!end || end->isBeforeInBlock(other.end)) {
      end = other.end;
    }
  }
};

// Analysis cache for avoiding redundant dataflow analysis
struct AnalysisCache {
  DenseMap<Operation *, const lwe::KeyStateLattice *> keyStateLattices;
  bool isValid = false;

  void invalidate() {
    keyStateLattices.clear();
    isValid = false;
  }

  void cacheResult(Operation *op, const lwe::KeyStateLattice *lattice) {
    if (op && lattice) {
      keyStateLattices[op] = lattice;
    }
  }

  const lwe::KeyStateLattice *getCachedResult(Operation *op) {
    auto it = keyStateLattices.find(op);
    return it != keyStateLattices.end() ? it->second : nullptr;
  }

  void removeOperation(Operation *op) { keyStateLattices.erase(op); }
};

// Structure to hold optimization results with distance stats
struct PairOptimizationWithStats {
  Operation *firstDeser = nullptr;
  Operation *firstClear = nullptr;
  int realDistance = 0;
  int totalDistance = 0;

  bool isValid() const { return firstDeser && firstClear; }
};

struct IncrementalMergeEvalKeys
    : impl::IncrementalMergeEvalKeysBase<IncrementalMergeEvalKeys> {
 private:
  AnalysisCache analysisCache;
  SmallVector<WorkRange> dirtyRanges;

  // Mark a range as needing re-analysis
  void markRangeDirty(Operation *start, Operation *end) {
    if (!start || !end) return;

    // Ensure operations are in the same block
    if (start->getBlock() != end->getBlock()) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Warning: markRangeDirty called with ops from different blocks\n");
      return;
    }

    // Ensure start comes before end
    if (end->isBeforeInBlock(start)) {
      std::swap(start, end);
    }

    WorkRange newRange(start, end);

    // Merge overlapping ranges
    auto it = dirtyRanges.begin();
    while (it != dirtyRanges.end()) {
      if (it->overlaps(newRange)) {
        newRange.merge(*it);
        it = dirtyRanges.erase(it);
      } else {
        ++it;
      }
    }

    dirtyRanges.push_back(newRange);

    if (enableDebug) {
      LLVM_DEBUG(llvm::dbgs() << "Marked dirty range: " << start->getLoc()
                              << " to " << end->getLoc() << "\n");
    }
  }

  // Check if an operation is in a dirty range
  bool isInDirtyRange(Operation *op) {
    for (const auto &range : dirtyRanges) {
      if (range.contains(op)) {
        return true;
      }
    }
    return false;
  }

  // Get analysis result with caching
  const lwe::KeyStateLattice *getKeyStateLattice(Operation *op,
                                                 DataFlowSolver &solver) {
    // If operation is in a dirty range, always re-compute
    if (isInDirtyRange(op)) {
      const auto *lattice =
          solver.lookupState<lwe::KeyStateLattice>(op->getResult(0));
      analysisCache.cacheResult(op, lattice);
      return lattice;
    }

    // Try to use cached result
    if (analysisCache.isValid) {
      const auto *cached = analysisCache.getCachedResult(op);
      if (cached) {
        return cached;
      }
    }

    // Fallback to fresh analysis
    const auto *lattice =
        solver.lookupState<lwe::KeyStateLattice>(op->getResult(0));
    analysisCache.cacheResult(op, lattice);
    return lattice;
  }

  // Helper to check if an operation is a key management operation
  bool isKeyManagementOp(Operation *op) {
    return isa<openfhe::DeserializeKeyOp, openfhe::ClearKeyOp>(op);
  }

  // Find optimizable pairs within distance threshold
  PairOptimizationWithStats findOptimizablePairWithDistance(
      openfhe::DeserializeKeyOp deserOp, int64_t keyIndex,
      DataFlowSolver &solver) {
    PairOptimizationWithStats result;
    Operation *currentOp = deserOp;
    int realDistance = 0;
    int totalDistance = 0;

    // Find the clear op associated with our deserialize
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
           realDistance < static_cast<int>(maxMergeDistance)) {
      totalDistance++;
      if (!isKeyManagementOp(currentOp)) {
        realDistance++;
      }

      if (auto existingDeser = dyn_cast<openfhe::DeserializeKeyOp>(currentOp)) {
        const auto *lattice = getKeyStateLattice(existingDeser, solver);

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
          if (firstClear && deserOp &&
              firstClear->getBlock() == deserOp->getBlock() &&
              firstClear->isBeforeInBlock(deserOp)) {
            result.firstDeser = existingDeser;
            result.firstClear = firstClear;
            result.realDistance = realDistance;
            result.totalDistance = totalDistance;

            if (enableDebug) {
              LLVM_DEBUG(llvm::dbgs()
                         << "Found optimal pair for key index " << keyIndex
                         << " with real distance " << realDistance
                         << " (total ops: " << totalDistance << ")\n");
            }
            return result;
          }
        }
      }
    }
    return result;
  }

 public:
  using IncrementalMergeEvalKeysBase::IncrementalMergeEvalKeysBase;

  void runOnOperation() override {
    bool madeChanges = true;
    unsigned iterationCount = 0;

    // Initialize: mark entire program as dirty for first iteration
    // Walk through all blocks and mark them as dirty
    getOperation()->walk([&](Block *block) {
      if (block->empty()) return;

      Operation *firstOp = &block->front();
      Operation *lastOp = &block->back();

      if (firstOp && lastOp) {
        markRangeDirty(firstOp, lastOp);
      }
    });

    LLVM_DEBUG(llvm::dbgs()
               << "Starting IncrementalMergeEvalKeys with max distance: "
               << maxMergeDistance << ", max iterations: " << maxIterations
               << "\n");

    while (madeChanges && iterationCount < maxIterations) {
      madeChanges = false;
      iterationCount++;

      // Only run analysis if we have dirty ranges
      if (dirtyRanges.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "No dirty ranges, stopping early\n");
        break;
      }

      if (enableDebug) {
        LLVM_DEBUG(llvm::dbgs() << "Iteration " << iterationCount << " with "
                                << dirtyRanges.size() << " dirty ranges\n");
      }

      DataFlowSolver solver;
      solver.load<dataflow::DeadCodeAnalysis>();
      solver.load<dataflow::SparseConstantPropagation>();
      solver.load<lwe::KeyStateAnalysis>();

      if (failed(solver.initializeAndRun(getOperation()))) {
        return signalPassFailure();
      }

      SmallVector<Operation *> opsToDelete;
      SmallVector<WorkRange> newDirtyRanges;

      auto walkResult =
          getOperation()->walk([&](openfhe::DeserializeKeyOp deserOp) {
            // Skip if not in dirty range (no changes needed)
            if (!isInDirtyRange(deserOp)) {
              return WalkResult::advance();
            }

            const auto *lattice = getKeyStateLattice(deserOp, solver);
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

            // Handle key depth matching
            if (deserOp->hasAttr("key_depth") &&
                firstDeserOp->hasAttr("key_depth")) {
              auto firstLevel =
                  firstDeserOp->getAttrOfType<IntegerAttr>("key_depth")
                      .getInt();
              auto secondLevel =
                  deserOp->getAttrOfType<IntegerAttr>("key_depth").getInt();
              if (firstLevel != secondLevel) {
                return WalkResult::advance();
              }
            } else if (!firstDeserOp->hasAttr("key_depth") &&
                       deserOp->hasAttr("key_depth")) {
              firstDeserOp->setAttr("key_depth", deserOp->getAttr("key_depth"));
            }

            // Replace uses of the second deserialize with the first one
            deserOp.getResult().replaceAllUsesWith(
                optimization.firstDeser->getResult(0));

            // Track the operations we're deleting
            opsToDelete.push_back(deserOp);
            opsToDelete.push_back(optimization.firstClear);

            // Mark the affected range as dirty for next iteration
            // Range from the first deserialize to some operations after the
            // second deserialize
            Operation *rangeStart = optimization.firstDeser;
            Operation *rangeEnd = deserOp;

            // Extend range to include a buffer for potential future merges
            for (unsigned i = 0;
                 i < maxMergeDistance && rangeEnd && rangeEnd->getNextNode();
                 i++) {
              rangeEnd = rangeEnd->getNextNode();
            }

            // Ensure we have a valid range in the same block
            if (rangeStart && rangeEnd &&
                rangeStart->getBlock() == rangeEnd->getBlock()) {
              newDirtyRanges.emplace_back(rangeStart, rangeEnd);
            }

            madeChanges = true;
            return WalkResult::advance();
          });

      if (walkResult.wasInterrupted()) {
        return signalPassFailure();
      }

      // Delete marked ops and invalidate cache for affected regions
      for (Operation *op : opsToDelete) {
        analysisCache.removeOperation(op);
        op->erase();
      }

      // Update dirty ranges for next iteration
      dirtyRanges.clear();
      for (const auto &range : newDirtyRanges) {
        markRangeDirty(range.start, range.end);
      }

      // Mark cache as valid for regions that weren't modified
      if (!madeChanges) {
        analysisCache.isValid = true;
      }
    }

    LLVM_DEBUG(llvm::dbgs()
               << "IncrementalMergeEvalKeys: Completed in " << iterationCount
               << " iterations with incremental analysis\n");
  }
};

}  // namespace heir
}  // namespace mlir
