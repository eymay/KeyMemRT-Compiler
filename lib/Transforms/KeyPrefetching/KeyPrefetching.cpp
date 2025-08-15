// Super Simple KeyPrefetching.cpp
// 1. Create enqueue for every deserialize at standard distance
// 2. Move ALL enqueues back by 10 operations

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

// Simple operation cost
inline int64_t getOperationCost(Operation* op) {
  return llvm::TypeSwitch<Operation*, int64_t>(op)
      .Case<openfhe::AddOp, openfhe::SubOp, openfhe::AddPlainOp,
            openfhe::SubPlainOp>([](auto) { return 1; })
      .Case<openfhe::MulOp, openfhe::MulNoRelinOp>([](auto) { return 11; })
      .Case<openfhe::MulPlainOp>([](auto) { return 2; })
      .Case<openfhe::ChebyshevOp>([](auto) { return 104; })
      .Case<openfhe::BootstrapOp>([](auto) { return 942; })
      .Case<openfhe::RotOp, openfhe::AutomorphOp>([](auto) { return 10; })
      .Case<openfhe::RelinOp>([](auto) { return 8; })
      .Case<openfhe::ModReduceOp, openfhe::LevelReduceOp>(
          [](auto) { return 3; })
      .Case<openfhe::DeserializeKeyOp, openfhe::SerializeKeyOp,
            openfhe::ClearKeyOp, openfhe::EnqueueKeyOp, openfhe::ClearCtOp,
            openfhe::CompressKeyOp>([](auto) { return 0; })
      .Default([](Operation*) { return 1; });
}

// Simple operation distance - 1 for OpenFHE ops, 0 for everything else
inline int getOperationDistance(Operation* op) {
  return llvm::TypeSwitch<Operation*, int>(op)
      // All OpenFHE computational operations count as 1
      .Case<openfhe::AddOp, openfhe::SubOp, openfhe::AddPlainOp,
            openfhe::SubPlainOp>([](auto) { return 1; })
      .Case<openfhe::MulOp, openfhe::MulNoRelinOp, openfhe::MulPlainOp>(
          [](auto) { return 1; })
      .Case<openfhe::ChebyshevOp, openfhe::BootstrapOp>([](auto) { return 1; })
      .Case<openfhe::RotOp, openfhe::AutomorphOp>([](auto) { return 1; })
      .Case<openfhe::RelinOp>([](auto) { return 1; })
      .Case<openfhe::ModReduceOp, openfhe::LevelReduceOp>(
          [](auto) { return 1; })
      // OpenFHE key/memory operations also count as 1
      .Case<openfhe::DeserializeKeyOp, openfhe::SerializeKeyOp,
            openfhe::ClearKeyOp, openfhe::EnqueueKeyOp, openfhe::ClearCtOp,
            openfhe::CompressKeyOp>([](auto) { return 1; })
      // Everything else (assignments, logging, etc.) counts as 0
      .Default([](Operation*) { return 0; });
}

struct KeyPrefetching : impl::KeyPrefetchingBase<KeyPrefetching> {
  using impl::KeyPrefetchingBase<KeyPrefetching>::KeyPrefetchingBase;

  void runOnOperation() override {
    Operation* op = getOperation();
    MLIRContext* ctx = op->getContext();

    LLVM_DEBUG(llvm::dbgs()
               << "KeyPrefetching: Starting with global enqueue option\n");

    // Check if global enqueue mode is enabled
    if (globalEnqueue) {
      LLVM_DEBUG(llvm::dbgs() << "KeyPrefetching: Using global enqueue mode\n");
      createGlobalEnqueues(op, ctx);
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "KeyPrefetching: Using standard positioning mode\n");

      // Step 1: Create enqueue for every deserialize
      createAllEnqueues(op, ctx);

      // Step 2: Move all enqueues back by 25 operations
      moveAllEnqueuesBack(op);

      // Step 3: Special handling for bootstrap keys
      handleBootstrapKeys(op);
    }

    LLVM_DEBUG(llvm::dbgs() << "KeyPrefetching: Completed approach\n");
  }

 private:
  // Global enqueue mode: Create all enqueues at the top of the program
  void createGlobalEnqueues(Operation* funcOp, MLIRContext* ctx) {
    // Collect all deserialize operations in program order
    SmallVector<openfhe::DeserializeKeyOp> deserializeOps;
    funcOp->walk([&](openfhe::DeserializeKeyOp deserOp) {
      deserializeOps.push_back(deserOp);
    });

    LLVM_DEBUG(llvm::dbgs() << "KeyPrefetching: Found " << deserializeOps.size()
                            << " deserialize operations for global enqueue\n");

    if (deserializeOps.empty()) {
      return;
    }

    // Find the insertion point at the top of the program
    Operation* insertionPoint = findTopOfProgram(funcOp);
    if (!insertionPoint) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "KeyPrefetching: Could not find top of program insertion point\n");
      return;
    }

    // Create enqueues in the EXACT same order as deserializes appear in the
    // program NO deduplication - if there are 3 deserializes for key 13, create
    // 3 enqueues for key 13
    OpBuilder builder(insertionPoint);
    unsigned enqueueCount = 0;

    for (auto deserOp : deserializeOps) {
      auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index");
      if (!indexAttr) {
        LLVM_DEBUG(llvm::dbgs() << "KeyPrefetching: Skipping deserialize "
                                   "without index attribute\n");
        continue;
      }

      int64_t rotIndex = indexAttr.getInt();
      unsigned depth = getDeserializeDepth(deserOp);

      // Create global enqueue at the top - ALWAYS, even if duplicate
      auto cryptoContext = deserOp.getCryptoContext();
      auto depthAttr = deserOp->getAttrOfType<IntegerAttr>("key_depth");
      if (!depthAttr) {
        depthAttr = deserOp->getAttrOfType<IntegerAttr>("depth");
      }

      builder.create<openfhe::EnqueueKeyOp>(deserOp.getLoc(), cryptoContext,
                                            indexAttr, depthAttr);

      enqueueCount++;
      LLVM_DEBUG(llvm::dbgs()
                 << "KeyPrefetching: Created global enqueue #" << enqueueCount
                 << " for key (" << rotIndex << "," << depth << ")\n");
    }

    LLVM_DEBUG(
        llvm::dbgs()
        << "KeyPrefetching: Created " << enqueueCount
        << " global enqueues at top of program (including duplicates)\n");
  }

  // Find the top of the program for global enqueue insertion
  Operation* findTopOfProgram(Operation* funcOp) {
    // Look for the first "real" operation after crypto context setup
    Operation* insertionPoint = nullptr;

    funcOp->walk([&](Operation* op) {
      // Found the first real operation - insert before it
      if (!insertionPoint) {
        insertionPoint = op;
      }
    });

    // If no specific insertion point found, use the first operation in the
    // function
    if (!insertionPoint) {
      for (auto& block : funcOp->getRegions().front()) {
        if (!block.empty()) {
          insertionPoint = &block.front();
          break;
        }
      }
    }

    return insertionPoint;
  }

  // Extract depth from deserialize operation (helper for global mode)
  unsigned getDeserializeDepth(openfhe::DeserializeKeyOp deserOp) {
    if (auto depthAttr = deserOp->getAttrOfType<IntegerAttr>("key_depth")) {
      return std::max(0U, static_cast<unsigned>(depthAttr.getInt()));
    }
    if (auto depthAttr = deserOp->getAttrOfType<IntegerAttr>("depth")) {
      return std::max(0U, static_cast<unsigned>(depthAttr.getInt()));
    }
    return 0;
  }
  // Step 1: Create enqueue for every deserialize at standard distance
  void createAllEnqueues(Operation* funcOp, MLIRContext* ctx) {
    SmallVector<openfhe::DeserializeKeyOp> deserializeOps;
    funcOp->walk([&](openfhe::DeserializeKeyOp deserOp) {
      deserializeOps.push_back(deserOp);
    });

    LLVM_DEBUG(llvm::dbgs() << "KeyPrefetching: Found " << deserializeOps.size()
                            << " deserialize operations\n");

    for (auto deserOp : deserializeOps) {
      createSimpleEnqueue(deserOp, ctx);
    }
  }

  // Step 2: Move all enqueues back by 25 operations
  void moveAllEnqueuesBack(Operation* funcOp) {
    SmallVector<openfhe::EnqueueKeyOp> enqueueOps;
    funcOp->walk([&](openfhe::EnqueueKeyOp enqueueOp) {
      enqueueOps.push_back(enqueueOp);
    });

    LLVM_DEBUG(llvm::dbgs() << "KeyPrefetching: Moving " << enqueueOps.size()
                            << " enqueues back by 25 operations\n");

    for (auto enqueueOp : enqueueOps) {
      moveEnqueueBack(enqueueOp.getOperation(), 25);
    }
  }

  // Step 3: Special handling for bootstrap keys
  void handleBootstrapKeys(Operation* funcOp) {
    // Find bootstrap operations and extract rotation indices
    std::set<int32_t> bootstrapIndices = findBootstrapIndices(funcOp);

    if (bootstrapIndices.empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "KeyPrefetching: No bootstrap operations found\n");
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "KeyPrefetching: Found bootstrap indices: ");
    for (int32_t idx : bootstrapIndices) {
      LLVM_DEBUG(llvm::dbgs() << idx << " ");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");

    // Find all enqueue operations for bootstrap keys
    SmallVector<openfhe::EnqueueKeyOp> bootstrapEnqueues;
    funcOp->walk([&](openfhe::EnqueueKeyOp enqueueOp) {
      if (auto indexAttr = enqueueOp->getAttrOfType<IntegerAttr>("index")) {
        int32_t rotIndex = indexAttr.getInt();
        if (bootstrapIndices.count(rotIndex)) {
          bootstrapEnqueues.push_back(enqueueOp);
        }
      }
    });

    LLVM_DEBUG(llvm::dbgs()
               << "KeyPrefetching: Processing " << bootstrapEnqueues.size()
               << " bootstrap enqueues\n");

    // Process each bootstrap enqueue
    for (auto enqueueOp : bootstrapEnqueues) {
      handleBootstrapEnqueue(enqueueOp);
    }
  }

  // Find bootstrap rotation indices from BootstrapOp attributes
  std::set<int32_t> findBootstrapIndices(Operation* funcOp) {
    std::set<int32_t> indices;

    funcOp->walk([&](openfhe::BootstrapOp bootstrapOp) {
      if (auto rotationIndicesAttr =
              bootstrapOp->getAttrOfType<ArrayAttr>("rotation_indices")) {
        for (auto indexAttr : rotationIndicesAttr) {
          if (auto intAttr = llvm::dyn_cast<IntegerAttr>(indexAttr)) {
            indices.insert(intAttr.getInt());
          }
        }
      }
    });

    return indices;
  }

  // Handle individual bootstrap enqueue
  void handleBootstrapEnqueue(openfhe::EnqueueKeyOp enqueueOp) {
    auto indexAttr = enqueueOp->getAttrOfType<IntegerAttr>("index");
    if (!indexAttr) return;

    int32_t rotIndex = indexAttr.getInt();
    LLVM_DEBUG(llvm::dbgs()
               << "KeyPrefetching: Handling bootstrap enqueue for key "
               << rotIndex << "\n");

    // Try to move further back by 10 operations, avoiding clear operations
    Operation* newPosition =
        findBootstrapPosition(enqueueOp.getOperation(), rotIndex);

    if (newPosition && newPosition != enqueueOp.getOperation()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "KeyPrefetching: Moving bootstrap enqueue for key "
                 << rotIndex << " to avoid conflicts\n");
      enqueueOp.getOperation()->moveBefore(newPosition);
    }
  }

  // Find optimal position for bootstrap enqueue, avoiding clear operations
  Operation* findBootstrapPosition(Operation* enqueueOp, int32_t rotIndex) {
    Operation* currentPos = enqueueOp;
    Operation* targetPos = currentPos;
    unsigned meaningfulOpsWalked = 0;

    // Try to walk back 10 more meaningful operations
    while (currentPos && meaningfulOpsWalked < 10) {
      currentPos = currentPos->getPrevNode();
      if (!currentPos) break;

      // Don't cross block boundaries
      if (currentPos->getBlock() != enqueueOp->getBlock()) {
        currentPos = currentPos->getNextNode();  // Back up one
        break;
      }

      // Check if this is a clear operation for our key
      if (auto clearOp = dyn_cast<openfhe::ClearKeyOp>(currentPos)) {
        int32_t clearRotIndex = inferRotationIndexFromClear(clearOp);
        if (clearRotIndex == rotIndex) {
          LLVM_DEBUG(llvm::dbgs()
                     << "KeyPrefetching: Found clear for bootstrap key "
                     << rotIndex << ", placing enqueue right after clear\n");
          // Place enqueue right after this clear operation
          return clearOp->getNextNode();
        }
      }

      // Count this operation if it's meaningful
      if (getOperationDistance(currentPos) > 0) {
        meaningfulOpsWalked++;
        targetPos = currentPos;
      }
    }

    // If no clear operation found, use the position 10 meaningful ops back
    return targetPos;
  }

  // Infer rotation index from clear operation (simplified)
  int32_t inferRotationIndexFromClear(openfhe::ClearKeyOp clearOp) {
    // Look backwards for recent deserialize operation to infer the key
    Operation* current = clearOp.getOperation();
    for (int i = 0; i < 5 && current; i++) {
      current = current->getPrevNode();
      if (!current) break;

      if (auto deserOp = dyn_cast<openfhe::DeserializeKeyOp>(current)) {
        if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
          return indexAttr.getInt();
        }
      }
    }
    return -1;  // Could not infer
  }

  // Create simple enqueue at standard threshold distance
  void createSimpleEnqueue(openfhe::DeserializeKeyOp deserOp,
                           MLIRContext* ctx) {
    auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index");
    if (!indexAttr) return;

    int64_t rotIndex = indexAttr.getInt();

    // Check for existing enqueue nearby
    if (hasNearbyEnqueue(deserOp, rotIndex)) {
      LLVM_DEBUG(llvm::dbgs() << "KeyPrefetching: Skipping key " << rotIndex
                              << " - duplicate nearby\n");
      return;
    }

    // Find standard placement
    Operation* placementPoint = findStandardPlacement(deserOp);
    if (!placementPoint) {
      LLVM_DEBUG(llvm::dbgs() << "KeyPrefetching: No placement found for key "
                              << rotIndex << "\n");
      return;
    }

    // Create enqueue
    OpBuilder builder(placementPoint);
    auto cryptoContext = deserOp.getCryptoContext();
    auto depthAttr = deserOp->getAttrOfType<IntegerAttr>("key_depth");
    if (!depthAttr) {
      depthAttr = deserOp->getAttrOfType<IntegerAttr>("depth");
    }

    builder.create<openfhe::EnqueueKeyOp>(deserOp.getLoc(), cryptoContext,
                                          indexAttr, depthAttr);

    LLVM_DEBUG(llvm::dbgs() << "KeyPrefetching: Created enqueue for key "
                            << rotIndex << "\n");
  }

  // Find standard placement at threshold distance
  Operation* findStandardPlacement(openfhe::DeserializeKeyOp deserOp) {
    Operation* current = deserOp.getOperation();
    Operation* placementPoint = current;
    int64_t accumulatedCost = 0;
    unsigned opsWalked = 0;

    // Walk backwards accumulating costs
    while (current && opsWalked < 200) {
      current = current->getPrevNode();
      if (!current) break;
      opsWalked++;

      int64_t opCost = getOperationCost(current);
      accumulatedCost += opCost;
      placementPoint = current;

      if (accumulatedCost >= prefetchThreshold.getValue()) {
        break;
      }
    }

    return placementPoint;
  }

  // Move an enqueue operation back by specified number of operations
  void moveEnqueueBack(Operation* enqueueOp, unsigned numOps) {
    Operation* newPosition = enqueueOp;
    unsigned opsWalked = 0;

    // Walk back counting only meaningful operations
    while (newPosition && opsWalked < numOps) {
      newPosition = newPosition->getPrevNode();
      if (!newPosition) break;

      // Don't cross block boundaries
      if (newPosition->getBlock() != enqueueOp->getBlock()) {
        newPosition = newPosition->getNextNode();  // Back up one
        break;
      }

      // Count this operation if it's meaningful (OpenFHE ops = 1, others = 0)
      opsWalked += getOperationDistance(newPosition);
    }

    // Move the enqueue if we found a valid new position
    if (newPosition && newPosition != enqueueOp) {
      LLVM_DEBUG(llvm::dbgs() << "KeyPrefetching: Moving enqueue back by "
                              << opsWalked << " meaningful operations\n");
      enqueueOp->moveBefore(newPosition);
    }
  }

  // Check if deserialize already has a nearby enqueue
  bool hasNearbyEnqueue(openfhe::DeserializeKeyOp deserOp, int64_t rotIndex) {
    Operation* current = deserOp.getOperation();
    for (int i = 0; i < 30 && current; i++) {
      current = current->getPrevNode();
      if (!current) break;

      if (auto existingEnqueue = dyn_cast<openfhe::EnqueueKeyOp>(current)) {
        if (auto existingIndexAttr =
                existingEnqueue->getAttrOfType<IntegerAttr>("index")) {
          if (existingIndexAttr.getInt() == rotIndex) {
            return true;
          }
        }
      }
    }
    return false;
  }
};

}  // namespace
}  // namespace heir
}  // namespace mlir
