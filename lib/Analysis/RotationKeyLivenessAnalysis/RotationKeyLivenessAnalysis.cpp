#include "lib/Analysis/RotationKeyLivenessAnalysis/RotationKeyLivenessAnalysis.h"

#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"   // from @llvm-project

using namespace mlir;
using namespace heir;

// Helper to get operations in order
SmallVector<Operation*> getAllOperationsInOrder(
    const DenseMap<int64_t, RotationKeyInterval>& intervals) {
  SmallVector<Operation*> ops;
  DenseSet<Operation*> seen;

  // Collect all unique operations from intervals
  for (const auto& [_, interval] : intervals) {
    for (tensor_ext::RotateOp rotateOp : interval.uses) {
      Operation* op = rotateOp.getOperation();
      if (seen.insert(op).second) {
        ops.push_back(op);
      }
    }
  }

  // Sort by program order
  llvm::sort(ops,
             [](Operation* a, Operation* b) { return a->isBeforeInBlock(b); });

  return ops;
}

void RotationKeyLivenessAnalysis::dump() const {
  llvm::dbgs() << "Rotation Key Liveness Analysis Results:\n";

  // First dump the intervals for constant rotation indices
  for (const auto& [index, interval] : rotationKeyIntervals) {
    llvm::dbgs() << "Index " << index << ":\n";
    llvm::dbgs() << "  First use: " << *interval.firstUse << "\n";
    llvm::dbgs() << "  Last use: " << *interval.lastUse << "\n";
    llvm::dbgs() << "  Total uses: " << interval.uses.size() << "\n";
    llvm::dbgs() << "  Uses:\n";
    for (tensor_ext::RotateOp rotateOp : interval.uses) {
      llvm::dbgs() << "    " << *rotateOp << "\n";
    }
    llvm::dbgs() << "\n";
  }

  // Then dump any rotations with dynamic/non-constant shifts
  if (!dynamicShiftOps.empty()) {
    llvm::dbgs() << "Dynamic shift rotations:\n";
    for (tensor_ext::RotateOp rotateOp : dynamicShiftOps) {
      llvm::dbgs() << "  " << *rotateOp << "\n";
    }
  }

  // Find overlapping intervals
  llvm::dbgs() << "\nInterval overlaps:\n";
  for (const auto& [index1, interval1] : rotationKeyIntervals) {
    for (const auto& [index2, interval2] : rotationKeyIntervals) {
      if (index1 >= index2) continue;  // Avoid duplicate reporting

      // Check if intervals overlap
      bool overlaps = !interval1.lastUse->isBeforeInBlock(interval2.firstUse) &&
                      !interval2.lastUse->isBeforeInBlock(interval1.firstUse);

      if (overlaps) {
        llvm::dbgs() << "  Indices " << index1 << " and " << index2
                     << " have overlapping lifetimes\n";
      }
    }
  }

  // Add memory statistics
  llvm::dbgs() << "\nMemory statistics:\n";
  llvm::dbgs() << "  Total number of rotation indices: "
               << rotationKeyIntervals.size() << "\n";

  // Find max concurrent live keys
  size_t maxLive = 0;
  for (auto* op : getAllOperationsInOrder(rotationKeyIntervals)) {
    size_t liveCurrent = 0;
    for (const auto& [index, _] : rotationKeyIntervals) {
      if (isKeyNeededAt(index, op)) {
        liveCurrent++;
      }
    }
    maxLive = std::max(maxLive, liveCurrent);
  }
  llvm::dbgs() << "  Maximum concurrent live rotation keys: " << maxLive
               << "\n";
}
