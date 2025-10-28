#include "lib/Dialect/KMRT/Transforms/MergeRotationKeys.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <vector>

#include "lib/Dialect/KMRT/IR/KMRTOps.h"
#include "lib/Dialect/KMRT/IR/KMRTTypes.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/IntegerSet.h"
#include "mlir/include/mlir/IR/IRMapping.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Support/LLVM.h"

#define DEBUG_TYPE "kmrt-merge-rotation-keys"

namespace mlir {
namespace heir {
namespace kmrt {

#define GEN_PASS_DEF_MERGEROTATIONKEYS
#include "lib/Dialect/KMRT/Transforms/Passes.h.inc"

// Structure to hold optimization results with distance stats
struct PairOptimizationWithStats {
  Operation *firstLoad = nullptr;   // First load to keep
  Operation *firstClear = nullptr;  // First clear to remove
  int realDistance = 0;             // Distance excluding key ops
  int totalDistance = 0;            // Total operation count

  bool isValid() const { return firstLoad && firstClear; }
};

// Helper structure to represent a rotation key identity
// This works with both static and dynamic rotation keys
struct RotationKeyIdentity {
  std::optional<int64_t> staticIndex;  // For static rotation keys
  Value dynamicIndexValue;             // For dynamic rotation keys

  // Check if this is a static key
  bool isStatic() const { return staticIndex.has_value(); }

  // Check if this is a dynamic key
  bool isDynamic() const { return static_cast<bool>(dynamicIndexValue); }

  // Compare two key identities
  bool matches(const RotationKeyIdentity &other) const {
    // Both static with same index
    if (isStatic() && other.isStatic()) {
      return staticIndex == other.staticIndex;
    }

    // Both dynamic with same index value (SSA value comparison)
    if (isDynamic() && other.isDynamic()) {
      return dynamicIndexValue == other.dynamicIndexValue;
    }

    return false;
  }

  // Extract key identity from a LoadKeyOp
  static std::optional<RotationKeyIdentity> fromLoadKeyOp(LoadKeyOp loadOp) {
    RotationKeyIdentity identity;

    // Check if the result type has a static rotation index
    auto rotKeyType = llvm::cast<RotKeyType>(loadOp.getRotKey().getType());
    if (rotKeyType.isStatic()) {
      identity.staticIndex = rotKeyType.getStaticIndex();
      return identity;
    }

    // Dynamic key - store the index operand value
    identity.dynamicIndexValue = loadOp.getIndex();
    return identity;
  }
};

struct MergeRotationKeys : impl::MergeRotationKeysBase<MergeRotationKeys> {
  void runOnOperation() override {
    // First pass: Check for constant-indexed keys that can be reused in affine loops
    // This must happen BEFORE the basic merging pass
    SmallVector<ClearKeyOp> clearOpsToTransform;
    getOperation()->walk([&](ClearKeyOp clearOp) {
      // Get the key being cleared
      Value key = clearOp.getRotKey();
      auto loadOp = key.getDefiningOp<LoadKeyOp>();
      if (!loadOp) return;

      // Extract constant index (either from static type or constant operand)
      auto maybeIndex = getConstantIndex(loadOp);
      if (!maybeIndex) return;

      int64_t staticIndex = *maybeIndex;

      // Search forward to find affine loops that might use this key
      Operation *searchOp = clearOp->getNextNode();
      int distance = 0;

      while (searchOp && distance < maxMergeDistance * 2) {
        if (auto affineLoop = dyn_cast<affine::AffineForOp>(searchOp)) {
          if (loopMightLoadKey(affineLoop, staticIndex)) {
            clearOpsToTransform.push_back(clearOp);
            return;
          }
        }

        if (!isKeyManagementOp(searchOp)) {
          distance++;
        }

        searchOp = searchOp->getNextNode();
      }
    });

    // Now perform the transformations outside the walk
    for (ClearKeyOp clearOp : clearOpsToTransform) {
      reorderLoopForKeyReuse(clearOp);
    }

    // Second pass: Check for constant-indexed keys after loops that can reuse loop keys
    SmallVector<LoadKeyOp> postLoopLoadsToTransform;
    getOperation()->walk([&](LoadKeyOp loadOp) {
      // Extract constant index (either from static type or constant operand)
      auto maybeIndex = getConstantIndex(loadOp);
      if (!maybeIndex) return;

      int64_t staticIndex = *maybeIndex;

      // Search backwards to find affine loops
      Operation *searchOp = loadOp->getPrevNode();
      int distance = 0;

      while (searchOp && distance < maxMergeDistance * 2) {
        if (auto affineLoop = dyn_cast<affine::AffineForOp>(searchOp)) {
          if (loopMightLoadKey(affineLoop, staticIndex)) {
            postLoopLoadsToTransform.push_back(loadOp);
            return;
          }
        }

        if (!isKeyManagementOp(searchOp)) {
          distance++;
        }

        searchOp = searchOp->getPrevNode();
      }
    });

    // Perform post-loop optimizations
    for (LoadKeyOp loadOp : postLoopLoadsToTransform) {
      reuseLoopKeyForPostLoad(loadOp);
    }

    // Third pass: Basic key merging within blocks
    bool madeChanges = true;
    unsigned iterationCount = 0;

    // Keep iterating until no more pairs can be merged
    while (madeChanges) {
      madeChanges = false;
      iterationCount++;

      // Track ops to delete to avoid iterator invalidation
      SmallVector<Operation *> opsToDelete;

      auto walkResult = getOperation()->walk([&](LoadKeyOp loadOp) {
        // Extract key identity from this load operation
        auto maybeIdentity = RotationKeyIdentity::fromLoadKeyOp(loadOp);
        if (!maybeIdentity) {
          return WalkResult::advance();
        }

        RotationKeyIdentity keyIdentity = *maybeIdentity;

        // Find optimizable pair
        PairOptimizationWithStats optimization =
            findOptimizablePairWithDistance(loadOp, keyIdentity);

        if (!optimization.isValid()) {
          return WalkResult::advance();
        }

        auto firstLoadOp = cast<LoadKeyOp>(optimization.firstLoad);

        // Handle key_depth attribute if present (for compression)
        if (loadOp->hasAttr("key_depth") && firstLoadOp->hasAttr("key_depth")) {
          auto firstLevel =
              firstLoadOp->getAttrOfType<IntegerAttr>("key_depth").getInt();
          auto secondLevel =
              loadOp->getAttrOfType<IntegerAttr>("key_depth").getInt();
          if (firstLevel != secondLevel) {
            return WalkResult::advance();  // Skip merging different levels
          }
        } else if (!firstLoadOp->hasAttr("key_depth") &&
                   loadOp->hasAttr("key_depth")) {
          firstLoadOp->setAttr("key_depth", loadOp->getAttr("key_depth"));
        }

        // Replace uses of the second load with the first one
        loadOp.getRotKey().replaceAllUsesWith(
            optimization.firstLoad->getResult(0));

        // Track the second load and first clear for deletion
        opsToDelete.push_back(loadOp);
        opsToDelete.push_back(optimization.firstClear);

        // Mark that we made changes
        madeChanges = true;

        LLVM_DEBUG(llvm::dbgs()
                   << "Merged rotation key pair with real distance "
                   << optimization.realDistance << " (total ops: "
                   << optimization.totalDistance << ")\n";);

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
    LLVM_DEBUG(llvm::dbgs() << "MergeRotationKeys: Completed in "
                            << iterationCount << " iterations\n");
  }

 private:
  // Maximum distance (in non-key-management operations) for merging
  // This threshold balances memory usage vs. deserialization overhead
  static constexpr int maxMergeDistance = 10;

  // Helper to check if an operation is a key management operation
  bool isKeyManagementOp(Operation *op) {
    return isa<LoadKeyOp, ClearKeyOp, PrefetchKeyOp, UseKeyOp, AssumeLoadedOp>(op);
  }

  // Helper to extract constant index value from a load_key operation
  // Returns the constant index if available, either from:
  // 1. Static rotation key type (e.g., !kmrt.rot_key<rotation_index = 5>)
  // 2. Constant operand (e.g., kmrt.load_key %c5)
  std::optional<int64_t> getConstantIndex(LoadKeyOp loadOp) {
    // First check if the key type has a static index
    auto rotKeyType = llvm::cast<RotKeyType>(loadOp.getRotKey().getType());
    if (rotKeyType.isStatic()) {
      return rotKeyType.getStaticIndex();
    }

    // For dynamic keys, check if the index operand is a constant
    Value indexValue = loadOp.getIndex();
    if (auto constOp = indexValue.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        return intAttr.getInt();
      }
    }

    return std::nullopt;
  }

  // Reorder affine loop to reuse a pre-loaded key (static or constant-indexed)
  bool reorderLoopForKeyReuse(ClearKeyOp clearOp) {
    // Get the key being cleared
    Value key = clearOp.getRotKey();

    // Find the load_key that produced this key
    auto loadOp = key.getDefiningOp<LoadKeyOp>();
    if (!loadOp) return false;

    // Extract the constant rotation index (either from static type or constant operand)
    auto maybeIndex = getConstantIndex(loadOp);
    if (!maybeIndex) return false;

    int64_t staticIndex = *maybeIndex;

    // Search forward from the clear_key to find affine loops
    Operation *searchOp = clearOp->getNextNode();
    int distance = 0;

    while (searchOp && distance < maxMergeDistance * 2) {
      // Check if this is an affine.for loop
      if (auto affineLoop = dyn_cast<affine::AffineForOp>(searchOp)) {
        // Check if the loop might load the same key
        if (loopMightLoadKey(affineLoop, staticIndex)) {
          LLVM_DEBUG(llvm::dbgs() << "Found affine loop that will reuse key "
                                  << staticIndex << ", peeling iteration\n");

          // Perform the loop transformation
          peelIterationForKeyReuse(loadOp, clearOp, affineLoop, staticIndex);
          return true;
        }
      }

      // Count distance only for non-key-management operations
      if (!isKeyManagementOp(searchOp)) {
        distance++;
      }

      searchOp = searchOp->getNextNode();
    }

    return false;
  }

  // Helper struct to hold info about a load in potentially nested loops
  struct NestedLoadInfo {
    LoadKeyOp loadOp;
    ClearKeyOp clearOp;
    SmallVector<affine::AffineForOp, 4> enclosingLoops;  // Innermost to outermost
    SmallVector<int64_t, 4> targetIndices;  // Target index for each loop
  };

  // Find a load_key operation in nested loops that matches the target index
  std::optional<NestedLoadInfo> findNestedLoadOp(affine::AffineForOp topLoop, int64_t targetIndex) {
    NestedLoadInfo info;

    // Walk the loop hierarchy to find the load
    topLoop.walk([&](LoadKeyOp loadOp) {
      Value indexValue = loadOp.getIndex();

      // Trace back through index_cast
      if (auto indexCastOp = indexValue.getDefiningOp<arith::IndexCastOp>()) {
        indexValue = indexCastOp.getIn();
      }

      // Collect all enclosing affine loops
      SmallVector<affine::AffineForOp, 4> loops;
      SmallVector<Value, 4> loopIVs;
      Operation *parent = loadOp->getParentOp();
      while (parent) {
        if (auto affineLoop = dyn_cast<affine::AffineForOp>(parent)) {
          loops.push_back(affineLoop);
          loopIVs.push_back(affineLoop.getInductionVar());
        }
        parent = parent->getParentOp();
      }

      // Check if this load uses ANY of the loop induction variables
      bool usesLoopIV = false;
      for (Value iv : loopIVs) {
        if (indexValue == iv) {
          usesLoopIV = true;
          break;
        }
      }

      if (!usesLoopIV || loops.empty()) {
        return WalkResult::advance();
      }

      // Find which loop IV is used and check if targetIndex is in range
      for (size_t i = 0; i < loops.size(); ++i) {
        if (indexValue == loops[i].getInductionVar()) {
          std::optional<int64_t> lb = loops[i].getConstantLowerBound();
          std::optional<int64_t> ub = loops[i].getConstantUpperBound();

          if (lb && ub && targetIndex >= *lb && targetIndex < *ub) {
            // Found a match! Now collect target indices for all enclosing loops
            info.loadOp = loadOp;
            info.enclosingLoops = loops;  // Already innermost to outermost

            // For the matching loop, target is targetIndex
            // For outer loops, we need to compute what iteration would give us targetIndex
            info.targetIndices.push_back(targetIndex);

            // For outer loops (if any), assume iteration 0 for now
            // This is a simplification - in BSGS, outer loop 0 with inner loop targetIndex
            for (size_t j = i + 1; j < loops.size(); ++j) {
              info.targetIndices.push_back(0);
            }

            // Find the clear operation
            for (Operation *user : loadOp.getRotKey().getUsers()) {
              if (auto clearOp = dyn_cast<ClearKeyOp>(user)) {
                info.clearOp = clearOp;
                break;
              }
            }

            return WalkResult::interrupt();
          }
        }
      }

      return WalkResult::advance();
    });

    if (info.loadOp && info.clearOp) {
      return info;
    }
    return std::nullopt;
  }

  // Remap loop to reuse a pre-loaded key with conditional load/clear
  void peelIterationForKeyReuse(LoadKeyOp preLoadOp, ClearKeyOp clearOp,
                                affine::AffineForOp loop, int64_t targetIndex) {
    OpBuilder builder(loop);
    Location loc = loop.getLoc();

    // Find the load_key operation that might be in nested loops
    auto maybeInfo = findNestedLoadOp(loop, targetIndex);
    if (!maybeInfo) {
      LLVM_DEBUG(llvm::dbgs() << "Could not find nested load for key " << targetIndex << "\n");
      return;
    }

    NestedLoadInfo info = *maybeInfo;
    LoadKeyOp loopLoadOp = info.loadOp;
    ClearKeyOp loopClearOp = info.clearOp;

    // Wrap load_key with affine.if checking all enclosing loop IVs
    builder.setInsertionPoint(loopLoadOp);

    // Collect all loop induction variables (innermost to outermost)
    SmallVector<Value, 4> ivValues;
    for (auto enclosingLoop : info.enclosingLoops) {
      ivValues.push_back(enclosingLoop.getInductionVar());
    }

    // Create affine condition: conjunction of (iv_i == targetIndex_i) for all loops
    // For example: (d0 == 0 && d1 == 1) becomes (d0 - 0 == 0 && d1 - 1 == 0)
    SmallVector<AffineExpr, 4> exprs;
    SmallVector<bool, 4> eqFlags;

    for (size_t i = 0; i < ivValues.size(); ++i) {
      auto dimExpr = builder.getAffineDimExpr(i);
      auto expr = dimExpr - info.targetIndices[i];
      exprs.push_back(expr);
      eqFlags.push_back(true);  // equality constraint
    }

    // Create integer set with multiple equality constraints (all must be true)
    auto intSet = IntegerSet::get(ivValues.size(), 0, exprs, eqFlags);

    // Create affine.if - return dynamic key type
    auto ifOp = builder.create<affine::AffineIfOp>(
        loc, loopLoadOp.getRotKey().getType(), intSet, ivValues,
        /*withElseRegion=*/true);

    // In the "then" region: use kmrt.use_key to reference the pre-loaded key
    builder.setInsertionPointToStart(ifOp.getThenBlock());
    auto useKeyOp = builder.create<UseKeyOp>(loc, loopLoadOp.getRotKey().getType(), preLoadOp.getRotKey());
    builder.create<affine::AffineYieldOp>(loc, useKeyOp.getRotKey());

    // In the "else" region: load normally
    builder.setInsertionPointToStart(ifOp.getElseBlock());
    Operation *clonedLoad = builder.clone(*loopLoadOp.getOperation());
    builder.create<affine::AffineYieldOp>(loc, clonedLoad->getResult(0));

    // Replace uses of original load with affine.if result
    loopLoadOp.getRotKey().replaceAllUsesWith(ifOp.getResult(0));
    loopLoadOp.erase();

    // Erase the original clear operation before the loop
    clearOp.erase();

    LLVM_DEBUG(llvm::dbgs() << "Peeled iteration for key " << targetIndex
                            << " across " << ivValues.size() << " nested loops\n");
  }

  // Reuse a key from inside a loop for a post-loop load
  // Strategy: Skip clearing the key in the loop when iv == staticIndex,
  // then the post-loop load will be merged by the basic merging pass
  bool reuseLoopKeyForPostLoad(LoadKeyOp postLoadOp) {
    // Extract constant index (either from static type or constant operand)
    auto maybeIndex = getConstantIndex(postLoadOp);
    if (!maybeIndex) return false;

    int64_t staticIndex = *maybeIndex;

    // Search backwards to find the affine loop
    Operation *searchOp = postLoadOp->getPrevNode();
    int distance = 0;

    while (searchOp && distance < maxMergeDistance * 2) {
      if (auto affineLoop = dyn_cast<affine::AffineForOp>(searchOp)) {
        if (loopMightLoadKey(affineLoop, staticIndex)) {
          LLVM_DEBUG(llvm::dbgs() << "Found affine loop that loads key "
                                  << staticIndex << ", skipping clear for reuse\n");

          // Find the load and clear operations in the loop
          // The load might be inside an affine.if if the first pass already transformed it
          Block *loopBody = affineLoop.getBody();
          ClearKeyOp loopClearOp;
          Value loopKeyValue;

          for (Operation &op : loopBody->without_terminator()) {
            // Check for direct load_key operations
            if (auto loadOp = dyn_cast<LoadKeyOp>(&op)) {
              Value indexValue = loadOp.getIndex();
              if (auto indexCastOp = indexValue.getDefiningOp<arith::IndexCastOp>()) {
                if (indexCastOp.getIn() == affineLoop.getInductionVar()) {
                  loopKeyValue = loadOp.getRotKey();
                }
              }
            }
            // Check for affine.if operations that might contain load_key
            if (auto affineIfOp = dyn_cast<affine::AffineIfOp>(&op)) {
              if (affineIfOp.getNumResults() > 0 &&
                  isa<RotKeyType>(affineIfOp.getResult(0).getType())) {
                loopKeyValue = affineIfOp.getResult(0);
              }
            }
            // Find the clear operation
            if (auto clearOp = dyn_cast<ClearKeyOp>(&op)) {
              if (loopKeyValue && clearOp.getRotKey() == loopKeyValue) {
                loopClearOp = clearOp;
              }
            }
          }

          if (!loopKeyValue || !loopClearOp) return false;

          // Wrap the clear_key with affine.if to skip clearing when iv == staticIndex
          // Strategy: if (iv == staticIndex) { skip } else { clear }
          OpBuilder builder(loopClearOp);
          Location loc = loopClearOp.getLoc();
          Value ivValue = affineLoop.getInductionVar();

          // Create affine condition: iv == staticIndex
          // Integer set: (d0 - staticIndex == 0)
          auto dimExpr = builder.getAffineDimExpr(0);
          auto expr = dimExpr - staticIndex;
          auto intSet = IntegerSet::get(1, 0, {expr}, {true});  // equality constraint

          // Create affine.if with else block
          auto ifOp = builder.create<affine::AffineIfOp>(
              loc, TypeRange{}, intSet, ValueRange{ivValue},
              /*withElseRegion=*/true);

          // Then block (iv == staticIndex): do nothing, just skip clearing
          builder.setInsertionPointToStart(ifOp.getThenBlock());
          // Empty then block - just yield immediately

          // Else block (iv != staticIndex): clear the key
          builder.setInsertionPointToStart(ifOp.getElseBlock());
          builder.clone(*loopClearOp.getOperation());

          // Erase the original clear
          loopClearOp.erase();

          // Now the key from iteration staticIndex will remain in memory
          // Replace the post-loop load with assume_loaded
          OpBuilder postLoopBuilder(postLoadOp);
          auto assumeLoadedOp = postLoopBuilder.create<AssumeLoadedOp>(
              postLoadOp.getLoc(),
              postLoadOp.getRotKey().getType(),
              postLoadOp.getIndex());

          postLoadOp.getRotKey().replaceAllUsesWith(assumeLoadedOp.getRotKey());
          postLoadOp.erase();

          LLVM_DEBUG(llvm::dbgs() << "Replaced post-loop load with assume_loaded for key "
                                  << staticIndex << "\n");

          return true;
        }
      }

      if (!isKeyManagementOp(searchOp)) {
        distance++;
      }

      searchOp = searchOp->getPrevNode();
    }

    return false;
  }

  // Check if an affine loop (or nested loops inside it) might load a specific rotation key index
  bool loopMightLoadKey(affine::AffineForOp loop, int64_t targetIndex) {
    // Walk the loop body looking for load_key operations
    bool found = false;
    loop.walk([&](LoadKeyOp loadOp) {
      // Check if this load uses a loop induction variable (from this loop or any nested loop)
      Value indexValue = loadOp.getIndex();

      // Trace back through index_cast if present
      if (auto indexCastOp = indexValue.getDefiningOp<arith::IndexCastOp>()) {
        indexValue = indexCastOp.getIn();
      }

      // Collect all enclosing affine loops for this load
      SmallVector<affine::AffineForOp, 4> enclosingLoops;
      Operation *parent = loadOp->getParentOp();
      while (parent) {
        if (auto affineLoop = dyn_cast<affine::AffineForOp>(parent)) {
          enclosingLoops.push_back(affineLoop);
        }
        parent = parent->getParentOp();
      }

      // Check if this load uses ANY enclosing loop's IV and if targetIndex is in range
      for (auto enclosingLoop : enclosingLoops) {
        if (indexValue == enclosingLoop.getInductionVar()) {
          std::optional<int64_t> lb = enclosingLoop.getConstantLowerBound();
          std::optional<int64_t> ub = enclosingLoop.getConstantUpperBound();

          if (lb && ub && targetIndex >= *lb && targetIndex < *ub) {
            found = true;
            return WalkResult::interrupt();
          }
        }
      }

      return WalkResult::advance();
    });

    return found;
  }

  // Find an optimizable pair of load/clear operations for the same key
  PairOptimizationWithStats findOptimizablePairWithDistance(
      LoadKeyOp loadOp, const RotationKeyIdentity &keyIdentity) {
    PairOptimizationWithStats result;
    Operation *currentOp = loadOp;

    // Track the clear op associated with our load
    Operation *secondClear = nullptr;
    for (Operation *user : loadOp.getRotKey().getUsers()) {
      if (auto clearOp = dyn_cast<ClearKeyOp>(user)) {
        secondClear = clearOp;
        break;
      }
    }
    if (!secondClear) return result;

    // Walk backwards looking for matching load/clear pair
    while ((currentOp = currentOp->getPrevNode())) {
      if (auto existingLoad = dyn_cast<LoadKeyOp>(currentOp)) {
        // Check if this load operation has the same key identity
        auto maybeExistingIdentity =
            RotationKeyIdentity::fromLoadKeyOp(existingLoad);
        if (!maybeExistingIdentity) continue;

        RotationKeyIdentity existingIdentity = *maybeExistingIdentity;

        // Check if the key identities match
        if (keyIdentity.matches(existingIdentity)) {
          // Find its associated clear op
          Operation *firstClear = nullptr;
          for (Operation *user : existingLoad.getRotKey().getUsers()) {
            if (auto clearOp = dyn_cast<ClearKeyOp>(user)) {
              firstClear = clearOp;
              break;
            }
          }

          if (!firstClear) continue;

          // Verify firstClear comes before our second load
          if (firstClear->isBeforeInBlock(loadOp)) {
            // Count operations between firstClear and loadOp
            int realDistance = 0;
            int totalOps = 0;

            Operation *walkOp = firstClear->getNextNode();
            while (walkOp && walkOp != loadOp) {
              totalOps++;
              if (!isKeyManagementOp(walkOp)) {
                realDistance++;
              }
              walkOp = walkOp->getNextNode();
            }

            // Check if within distance threshold
            if (realDistance <= maxMergeDistance) {
              result.firstLoad = existingLoad;
              result.firstClear = firstClear;
              result.realDistance = realDistance;
              result.totalDistance = totalOps;

              LLVM_DEBUG(llvm::dbgs()
                             << "Found optimal pair for rotation key"
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

}  // namespace kmrt
}  // namespace heir
}  // namespace mlir
