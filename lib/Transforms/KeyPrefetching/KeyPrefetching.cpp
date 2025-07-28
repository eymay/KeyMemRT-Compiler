#include "lib/Transforms/KeyPrefetching/KeyPrefetching.h"

#include <algorithm>
#include <vector>

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "key-prefetching"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_KEYPREFETCHING
#include "lib/Transforms/KeyPrefetching/KeyPrefetching.h.inc"

namespace {

// Operation cost table - relative timing units
inline int64_t getOperationCostTyped(Operation* op) {
  return llvm::TypeSwitch<Operation*, int64_t>(op)
      // Computational operations
      .Case<openfhe::AddOp, openfhe::SubOp, openfhe::AddPlainOp,
            openfhe::SubPlainOp>([](auto) { return 1; })
      .Case<openfhe::MulOp, openfhe::MulPlainOp, openfhe::MulNoRelinOp>(
          [](auto) { return 10; })
      .Case<openfhe::ChebyshevOp>([](auto) { return 50; })
      .Case<openfhe::BootstrapOp>([](auto) { return 100; })
      .Case<openfhe::RotOp, openfhe::AutomorphOp>([](auto) { return 15; })
      .Case<openfhe::RelinOp>([](auto) { return 8; })
      .Case<openfhe::ModReduceOp, openfhe::LevelReduceOp>(
          [](auto) { return 3; })
      .Case<openfhe::KeySwitchOp>([](auto) { return 12; })
      // Non-computational operations (0 cost)
      .Case<openfhe::DeserializeKeyOp, openfhe::SerializeKeyOp,
            openfhe::ClearKeyOp, openfhe::EnqueueKeyOp, openfhe::ClearCtOp,
            openfhe::CompressKeyOp>([](auto) { return 0; })
      // Default case
      .Default([](Operation*) { return 1; });
}

// Depth-aware cost scaling using deserialize op's depth attribute
inline int64_t getScaledOperationCost(Operation* op, unsigned depth = 1) {
  int64_t baseCost = getOperationCostTyped(op);

  // Higher depth levels are faster (inverse relationship)
  if (depth > 1 && baseCost > 0) {
    return std::max(static_cast<int64_t>(1),
                    baseCost / static_cast<int64_t>(depth));
  }

  return baseCost;
}

struct KeyPrefetching : impl::KeyPrefetchingBase<KeyPrefetching> {
  using impl::KeyPrefetchingBase<KeyPrefetching>::KeyPrefetchingBase;
  void runOnOperation() override {
    Operation* op = getOperation();
    MLIRContext* ctx = op->getContext();

    // Find all deserialize operations
    SmallVector<openfhe::DeserializeKeyOp> deserializeOps;
    op->walk([&](openfhe::DeserializeKeyOp deserOp) {
      deserializeOps.push_back(deserOp);
    });

    LLVM_DEBUG(llvm::dbgs() << "KeyPrefetching: Found " << deserializeOps.size()
                            << " deserialize operations in function\n");

    // Track which keys we've seen to detect duplicates
    std::map<int64_t, int> keyCount;
    for (auto deserOp : deserializeOps) {
      if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
        int64_t rotIndex = indexAttr.getInt();
        keyCount[rotIndex]++;
        LLVM_DEBUG(llvm::dbgs()
                   << "KeyPrefetching: Found deserialize for key " << rotIndex
                   << " (occurrence #" << keyCount[rotIndex] << ")\n");
      }
    }

    // Report duplicates
    for (const auto& [key, count] : keyCount) {
      if (count > 1) {
        LLVM_DEBUG(llvm::dbgs()
                   << "KeyPrefetching: WARNING - Key " << key << " has "
                   << count << " deserialize operations\n");
      }
    }

    // Process each deserialize operation
    for (auto deserOp : deserializeOps) {
      insertEnqueueForDeserialize(deserOp, ctx);
    }

    LLVM_DEBUG(llvm::dbgs()
               << "KeyPrefetching: Processed " << deserializeOps.size()
               << " deserialize operations\n");
  }

 private:
  // Extract depth from deserialize operation
  unsigned getDeserializeDepth(openfhe::DeserializeKeyOp deserOp) {
    if (auto depthAttr = deserOp->getAttrOfType<IntegerAttr>("key_depth")) {
      return std::max(1U, static_cast<unsigned>(depthAttr.getInt()));
    }
    if (auto depthAttr = deserOp->getAttrOfType<IntegerAttr>("depth")) {
      return std::max(1U, static_cast<unsigned>(depthAttr.getInt()));
    }
    return 1;  // Default depth
  }

  // Find optimal placement for enqueue operation with detailed debug info
  Operation* findEnqueuePlacement(openfhe::DeserializeKeyOp deserOp,
                                  int64_t& totalCost, int& opsWalked) {
    int64_t accumulatedCost = 0;
    Operation* current = deserOp.getOperation();
    Operation* placementPoint = current;  // Default to just before deserialize
    opsWalked = 0;

    // Get depth from deserialize operation for cost scaling
    unsigned depth = getDeserializeDepth(deserOp);

    // Walk backwards accumulating costs
    while (current && accumulatedCost < prefetchThreshold.getValue()) {
      current = current->getPrevNode();
      if (!current) break;

      // Get cost for this operation (scaled by depth)
      int64_t opCost = getScaledOperationCost(current, depth);

      // Skip operations with 0 cost (non-computational) but still count them
      if (opCost == 0) {
        opsWalked++;
        continue;
      }

      accumulatedCost += opCost;
      opsWalked++;

      // Check if we've reached our threshold
      if (accumulatedCost >= prefetchThreshold.getValue()) {
        placementPoint = current;
        break;
      }

      placementPoint = current;  // Update placement point
    }

    totalCost = accumulatedCost;
    return placementPoint;
  }

  // Insert enqueue operation for a deserialize
  void insertEnqueueForDeserialize(openfhe::DeserializeKeyOp deserOp,
                                   MLIRContext* ctx) {
    // Extract attributes from deserialize operation first
    auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index");
    if (!indexAttr) {
      LLVM_DEBUG(llvm::dbgs() << "KeyPrefetching: Warning - DeserializeKeyOp "
                                 "missing index attribute\n");
      return;
    }

    int64_t rotIndex = indexAttr.getInt();

    // Check if there's already an enqueue operation for this key nearby
    Operation* current = deserOp.getOperation();
    for (int i = 0; i < 20 && current; i++) {  // Check 20 operations before
      current = current->getPrevNode();
      if (!current) break;

      if (auto existingEnqueue = dyn_cast<openfhe::EnqueueKeyOp>(current)) {
        if (auto existingIndexAttr =
                existingEnqueue->getAttrOfType<IntegerAttr>("index")) {
          if (existingIndexAttr.getInt() == rotIndex) {
            LLVM_DEBUG(llvm::dbgs()
                       << "KeyPrefetching: Skipping key " << rotIndex
                       << " - enqueue already exists nearby\n");
            return;  // Skip this one, already has enqueue
          }
        }
      }
    }

    // Find where to place the enqueue operation
    int64_t totalCost = 0;
    int opsWalked = 0;
    Operation* placementPoint =
        findEnqueuePlacement(deserOp, totalCost, opsWalked);

    // Create builder at placement point
    OpBuilder builder(placementPoint);

    // Extract attributes from deserialize operation
    auto cryptoContext = deserOp.getCryptoContext();
    auto depthAttr = deserOp->getAttrOfType<IntegerAttr>("key_depth");
    if (!depthAttr) {
      depthAttr = deserOp->getAttrOfType<IntegerAttr>("depth");
    }

    // Create enqueue operation with same parameters
    auto enqueueOp = builder.create<openfhe::EnqueueKeyOp>(
        deserOp.getLoc(), cryptoContext, indexAttr,
        depthAttr  // Pass depth as optional third parameter
    );

    // Copy depth attribute if present (remove manual setting since it's passed
    // to builder) The depth attribute is already passed in the create call
    // above

    unsigned depth = getDeserializeDepth(deserOp);

    // Debug statement for calculation in one line
    LLVM_DEBUG(llvm::dbgs()
               << "KeyPrefetching: key=" << rotIndex << " depth=" << depth
               << " walked=" << opsWalked << "ops"
               << " cost=" << totalCost
               << " threshold=" << prefetchThreshold.getValue() << " placed="
               << (totalCost >= prefetchThreshold.getValue() ? "threshold"
                                                             : "early")
               << "\n");
  }
};

}  // namespace

}  // namespace heir
}  // namespace mlir
