#include "lib/Dialect/KMRT/Transforms/MergeRotationKeys.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <vector>

#include "lib/Dialect/KMRT/IR/KMRTOps.h"
#include "lib/Dialect/KMRT/IR/KMRTTypes.h"
#include "lib/Dialect/KMRT/Transforms/AffineSetAnalysis.h"
#include "lib/Dialect/KMRT/Transforms/RotationKeyLivenessDFA.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/IRMapping.h"
#include "mlir/include/mlir/IR/IntegerSet.h"
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
  using MergeRotationKeysBase::MergeRotationKeysBase;

  void runOnOperation() override {
    llvm::errs() << "\n=== MergeRotationKeys: DFA-Based Analysis ===\n";

    // Step 1: Run dataflow analysis to track key liveness
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<kmrt::RotationKeyLivenessDFA>();

    if (failed(solver.initializeAndRun(getOperation()))) {
      llvm::errs() << "DFA solver failed\n";
      return signalPassFailure();
    }

    llvm::errs() << "DFA analysis completed successfully\n";

    // Step 2: Merge sequential clear-load pairs (e.g., between bootstraps)
    llvm::errs() << "=== Step 2: Sequential Clear-Load Merging ===\n";
    mergeSequentialClearLoadPairs(solver);

    // Step 3: Find preloop keys that can be merged with loops
    // Map: loop -> vector of (preloop key index, preloadOp, clearOp)
    DenseMap<Operation *,
             SmallVector<std::tuple<int64_t, LoadKeyOp, ClearKeyOp>>>
        loopToPreloopKeys;

    getOperation()->walk([&](ClearKeyOp clearOp) {
      // Get the key being cleared
      Value key = clearOp.getRotKey();
      auto loadOp = key.getDefiningOp<LoadKeyOp>();
      if (!loadOp) return;

      // Use DFA to check key state
      const auto *lattice = solver.lookupState<kmrt::RotationKeyLattice>(key);
      if (!lattice || !lattice->getValue().isLoaded()) {
        llvm::errs() << "  Key not in loaded state, skipping\n";
        return;
      }

      // Extract constant index
      auto maybeIndex = getConstantIndex(loadOp);
      if (!maybeIndex) return;

      int64_t preloopKeyIndex = *maybeIndex;
      // llvm::errs() << "  Found preloop key " << preloopKeyIndex << "\n";

      // Search forward for affine loops
      Operation *searchOp = clearOp->getNextNode();
      int distance = 0;

      while (searchOp && distance < 20) {
        if (auto affineLoop = dyn_cast<affine::AffineForOp>(searchOp)) {
          // Use affine set analysis to check intersection
          auto mergeOpp =
              kmrt::analyzeMergeOpportunity(preloopKeyIndex, affineLoop);

          if (mergeOpp && mergeOpp->isValid()) {
            // llvm::errs() << "  Found merge opportunity: key " <<
            // preloopKeyIndex
            //              << " can merge with loop\n";
            loopToPreloopKeys[affineLoop.getOperation()].push_back(
                std::make_tuple(preloopKeyIndex, loadOp, clearOp));
            return;
          }
        }

        if (!isa<LoadKeyOp, ClearKeyOp, UseKeyOp, AssumeLoadedOp>(searchOp)) {
          distance++;
        }

        searchOp = searchOp->getNextNode();
      }
    });

    // Step 4: Transform loops with preloop merge opportunities
    llvm::errs() << "Found " << loopToPreloopKeys.size()
                 << " loops with merge opportunities\n";

    for (auto &[loopOp, preloopKeys] : loopToPreloopKeys) {
      auto affineLoop = cast<affine::AffineForOp>(loopOp);

      llvm::errs() << "  Transforming loop with " << preloopKeys.size()
                   << " preloop keys: ";
      for (auto &[keyIndex, loadOp, clearOp] : preloopKeys) {
        llvm::errs() << keyIndex << " ";
      }
      llvm::errs() << "\n";

      // First try to extend existing affine.if (BSGS case)
      extendAffineIfForPreloopKeys(affineLoop, preloopKeys, solver);

      // If no affine.if was found, create new ones (standalone case)
      createStandaloneMerge(affineLoop, preloopKeys, solver);
    }

    // Step 4.5: Find postloop keys that can be merged with loops
    llvm::errs() << "=== Step 4.5: Postloop Merging ===\n";
    // First handle simple direct-load patterns
    findAndMergeSimplePostloopKeys(solver);
    // Then handle memref-based bootstrap patterns
    findAndMergePostloopKeys(solver);

    // Step 5: Apply loop peeling optimization for hot inner loops
    if (enableLoopPeeling) {
      llvm::errs() << "=== Step 5: Loop Peeling Optimization ===\n";
      SmallVector<affine::AffineForOp> outerLoops;
      getOperation()->walk([&](affine::AffineForOp loop) {
        // Find outer loops that contain inner loops with key management
        loop.walk([&](affine::AffineForOp innerLoop) {
          if (innerLoop != loop) {
            outerLoops.push_back(loop);
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
      });

      for (auto outerLoop : outerLoops) {
        peelInnerLoopKeyManagement(outerLoop);
      }
    }

    llvm::errs() << "=== MergeRotationKeys: Completed ===\n";
  }

 private:
  // Count FHE operations inside an operation (recursively for loops)
  int64_t countFHEOps(Operation *op) {
    // Check if this is an FHE operation
    bool isFHEOp = false;
    if (op->getDialect()) {
      StringRef dialectName = op->getDialect()->getNamespace();
      isFHEOp = dialectName == "openfhe";
    }

    // For loops, count operations inside and multiply by iteration count
    if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
      int64_t iterCount =
          forOp.getConstantUpperBound() - forOp.getConstantLowerBound();
      int64_t opsInBody = 0;

      // Count operations in the loop body
      for (auto &bodyOp : forOp.getBody()->without_terminator()) {
        opsInBody += countFHEOps(&bodyOp);
      }

      return iterCount * opsInBody;
    }

    // For other operations with regions, recurse into them
    if (op->getNumRegions() > 0) {
      int64_t totalOps = isFHEOp ? 1 : 0;
      for (Region &region : op->getRegions()) {
        for (Block &block : region.getBlocks()) {
          for (Operation &innerOp : block.without_terminator()) {
            totalOps += countFHEOps(&innerOp);
          }
        }
      }
      return totalOps;
    }

    // Leaf operation
    return isFHEOp ? 1 : 0;
  }

  // Merge sequential clear-load pairs for the same key
  void mergeSequentialClearLoadPairs(DataFlowSolver &solver) {
    SmallVector<std::pair<ClearKeyOp, LoadKeyOp>> pairsToMerge;

    getOperation()->walk([&](ClearKeyOp clearOp) {
      Value clearedKey = clearOp.getRotKey();
      auto clearLoadOp = clearedKey.getDefiningOp<LoadKeyOp>();
      if (!clearLoadOp) return;

      // Get the key index for this clear
      auto maybeClearIndex = getConstantIndex(clearLoadOp);
      if (!maybeClearIndex) return;

      int64_t clearKeyIndex = *maybeClearIndex;

      // Search forward for a load with the same key index
      Operation *searchOp = clearOp->getNextNode();
      int64_t distance = 0;
      const int64_t maxDistance =
          10;  // Allow much larger distance with loop unrolling

      while (searchOp && distance < maxDistance) {
        if (auto loadOp = dyn_cast<LoadKeyOp>(searchOp)) {
          auto maybeLoadIndex = getConstantIndex(loadOp);
          if (maybeLoadIndex && *maybeLoadIndex == clearKeyIndex) {
            // Found a matching load - mark for merging
            // llvm::errs() << "  Found clear-load pair for key " <<
            // clearKeyIndex
            //              << " at distance " << distance << " FHE ops\n";
            pairsToMerge.push_back({clearOp, loadOp});
            return;
          }
        }

        // Count FHE operations, accounting for loop bounds
        int64_t opCount = countFHEOps(searchOp);
        distance += opCount;

        searchOp = searchOp->getNextNode();
      }
    });

    llvm::errs() << "Found " << pairsToMerge.size()
                 << " clear-load pairs to merge\n";

    // Perform the merges
    for (auto &[clearOp, loadOp] : pairsToMerge) {
      Value originalKey = clearOp.getRotKey();
      // auto clearLoadOp = originalKey.getDefiningOp<LoadKeyOp>();
      // auto maybeClearIndex =
      //     clearLoadOp ? getConstantIndex(clearLoadOp) : std::nullopt;

      // llvm::errs() << "  Merging key "
      //              << (maybeClearIndex ? std::to_string(*maybeClearIndex) :
      //              "?")
      //              << ": removing clear and load\n";

      // Replace all uses of the loaded key with the original key
      loadOp.getRotKey().replaceAllUsesWith(originalKey);

      // Remove the load operation
      loadOp.erase();

      // Remove the clear operation
      clearOp.erase();
    }
  }

  // Helper to extract constant index from LoadKeyOp
  std::optional<int64_t> getConstantIndex(LoadKeyOp loadOp) {
    auto rotKeyType = llvm::cast<RotKeyType>(loadOp.getRotKey().getType());
    if (rotKeyType.isStatic()) {
      return rotKeyType.getStaticIndex();
    }

    Value indexValue = loadOp.getIndex();
    if (auto constOp = indexValue.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        return intAttr.getInt();
      }
    }

    return std::nullopt;
  }

  // Extend affine.if conditions in a loop to include preloop key merging
  // Find which memref index contains a specific key by analyzing stores
  std::optional<int64_t> findKeyInMemref(Value memref, int64_t targetKeyIndex) {
    // Build a mapping of key_index -> memref_index by analyzing stores
    DenseMap<int64_t, int64_t> keyToMemrefIndex;

    // Find all stores to this memref
    for (auto user : memref.getUsers()) {
      if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
        if (storeOp.getMemRef() != memref) continue;

        // Get the stored value (should be a rotation key)
        Value storedKey = storeOp.getValueToStore();

        // Trace back through any use_key operations to find the LoadKeyOp
        while (auto useKeyOp = storedKey.getDefiningOp<UseKeyOp>()) {
          storedKey = useKeyOp.getRotKey();
        }

        // Now check if we have a LoadKeyOp
        LoadKeyOp loadKeyOp = storedKey.getDefiningOp<LoadKeyOp>();
        if (!loadKeyOp) continue;

        // Extract the memref store index
        if (storeOp.getIndices().size() != 1) continue;
        Value storeIndexVal = storeOp.getIndices()[0];

        // Extract the key index
        auto maybeKeyIndex = getConstantIndex(loadKeyOp);

        // Case 1: Both load and store use constants
        if (maybeKeyIndex && storeIndexVal.getDefiningOp<arith::ConstantOp>()) {
          if (auto constOp = storeIndexVal.getDefiningOp<arith::ConstantOp>()) {
            if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
              int64_t memrefIdx = intAttr.getInt();
              int64_t keyIdx = *maybeKeyIndex;
              keyToMemrefIndex[keyIdx] = memrefIdx;
            }
          }
        }
        // Case 2: Loop IV pattern - load key %iv, store to memref[%iv]
        else if (auto blockArg = dyn_cast<BlockArgument>(storeIndexVal)) {
          // Check if the load index is the same loop IV
          if (loadKeyOp.getIndex() == storeIndexVal) {
            // Pattern: memref[i] = key[i]
            // Find the loop to get bounds
            if (auto forOp = dyn_cast<affine::AffineForOp>(
                    blockArg.getParentBlock()->getParentOp())) {
              int64_t lb = forOp.getConstantLowerBound();
              int64_t ub = forOp.getConstantUpperBound();
              // For all i in [lb, ub), key[i] is stored at memref[i]
              for (int64_t i = lb; i < ub; ++i) {
                keyToMemrefIndex[i] = i;
              }
            }
          }
        }
      }
    }

    // Look up the target key in our mapping
    auto it = keyToMemrefIndex.find(targetKeyIndex);
    if (it != keyToMemrefIndex.end()) {
      return it->second;
    }

    return std::nullopt;
  }

  // Find and merge postloop load-key operations with loop clear-key operations
  // Handle simple postloop pattern: loop that loads keys dynamically,
  // followed by a postloop load of the same key
  void findAndMergeSimplePostloopKeys(DataFlowSolver &solver) {
    getOperation()->walk([&](LoadKeyOp postloopLoad) {
      // Skip loads with key_depth (handled by findAndMergePostloopKeys)
      if (postloopLoad->hasAttr("key_depth")) {
        return;
      }

      // Extract constant index from postloop load
      auto maybeIndex = getConstantIndex(postloopLoad);
      if (!maybeIndex) return;
      int64_t postloopKeyIndex = *maybeIndex;

      // Search backward for a loop that loads this key
      Operation *searchOp = postloopLoad->getPrevNode();
      int distance = 0;

      while (searchOp && distance < 50) {
        if (auto affineLoop = dyn_cast<affine::AffineForOp>(searchOp)) {
          // Check if this loop loads keys with the loop IV
          LoadKeyOp loopLoad = nullptr;
          ClearKeyOp loopClear = nullptr;
          Value loopIV = affineLoop.getInductionVar();

          affineLoop.walk([&](LoadKeyOp loadOp) {
            // Check if this load uses the loop IV
            Value loadIndex = loadOp.getIndex();

            // Direct use of loop IV
            if (loadIndex == loopIV) {
              loopLoad = loadOp;
            }
            // Through index_cast
            else if (auto castOp =
                         loadIndex.getDefiningOp<arith::IndexCastOp>()) {
              if (castOp.getIn() == loopIV) {
                loopLoad = loadOp;
              }
            }
          });

          if (loopLoad) {
            // Check if the key is cleared in the loop
            affineLoop.walk([&](ClearKeyOp clearOp) {
              Value clearedKey = clearOp.getRotKey();
              // Trace through use_key operations
              while (auto useKey = clearedKey.getDefiningOp<UseKeyOp>()) {
                clearedKey = useKey.getRotKey();
              }
              // Check if it's clearing the loaded key
              if (clearedKey == loopLoad.getResult()) {
                loopClear = clearOp;
              }
            });
          }

          if (loopLoad && loopClear) {
            // Check if postloop key is within loop bounds
            int64_t lb = affineLoop.getConstantLowerBound();
            int64_t ub = affineLoop.getConstantUpperBound();

            if (postloopKeyIndex >= lb && postloopKeyIndex < ub) {
              llvm::errs()
                  << "    Found simple postloop pattern: loop loads keys " << lb
                  << " to " << ub << ", postloop loads key " << postloopKeyIndex
                  << "\n";

              // Wrap the clear with affine.if to skip postloop key
              OpBuilder builder(loopClear);
              Location loc = loopClear.getLoc();

              auto dimExpr = builder.getAffineDimExpr(0);
              auto condExpr = dimExpr - postloopKeyIndex;
              auto condSet = IntegerSet::get(1, 0, {condExpr}, {true});

              auto affineIf = builder.create<affine::AffineIfOp>(
                  loc, condSet, ValueRange{loopIV}, /*withElseRegion=*/true);

              // Move clear to else block
              loopClear->moveBefore(&affineIf.getElseRegion().front(),
                                    affineIf.getElseRegion().front().begin());

              // Replace postloop load with assume_loaded
              builder.setInsertionPoint(postloopLoad);
              auto assumeOp = builder.create<AssumeLoadedOp>(
                  postloopLoad.getLoc(), postloopLoad.getResult().getType(),
                  postloopLoad.getIndex());
              postloopLoad.getResult().replaceAllUsesWith(assumeOp.getResult());
              postloopLoad.erase();

              return;
            }
          }
        }

        if (!isa<LoadKeyOp, ClearKeyOp, UseKeyOp, AssumeLoadedOp>(searchOp)) {
          distance++;
        }
        searchOp = searchOp->getPrevNode();
      }
    });
  }

  void findAndMergePostloopKeys(DataFlowSolver &solver) {
    // Walk all LoadKeyOp operations to find postloop loads
    getOperation()->walk([&](LoadKeyOp loadOp) {
      // Only process loads with key_depth attribute (bootstrap keys)
      // This avoids interfering with regular BSGS baby step keys
      if (!loadOp->hasAttr("key_depth")) {
        return;
      }

      // Extract constant index from this load
      auto maybeIndex = getConstantIndex(loadOp);
      if (!maybeIndex) return;

      int64_t postloopKeyIndex = *maybeIndex;

      // Strategy 1: Search backward for any existing memref.load of this key
      // This is more general and can find loads from earlier in the function
      {
        Operation *searchOp = loadOp->getPrevNode();
        int distance = 0;

        while (searchOp && distance < 100) {
          // Skip loops - we'll handle them in strategy 2
          if (isa<affine::AffineForOp>(searchOp)) {
            searchOp = searchOp->getPrevNode();
            continue;
          }

          // Look for memref.load operations that load this key
          if (auto memLoad = dyn_cast<memref::LoadOp>(searchOp)) {
            // Check if this is a load with a constant index
            if (memLoad.getIndices().size() == 1) {
              if (auto constIdx = memLoad.getIndices()[0]
                                      .getDefiningOp<arith::ConstantOp>()) {
                if (auto intAttr = dyn_cast<IntegerAttr>(constIdx.getValue())) {
                  int64_t loadedIdx = intAttr.getInt();

                  // Check if the memref contains this key at this index
                  Value memref = memLoad.getMemRef();
                  auto memrefContainsKey =
                      findKeyInMemref(memref, postloopKeyIndex);

                  if (memrefContainsKey && *memrefContainsKey == loadedIdx) {
                    // Found an existing load of this key from a memref!
                    // Now search forward to find a loop that clears this memref
                    Operation *clearSearchOp = memLoad->getNextNode();
                    int clearDist = 0;

                    while (clearSearchOp && clearDist < 50) {
                      if (auto clearLoop =
                              dyn_cast<affine::AffineForOp>(clearSearchOp)) {
                        // Check if this loop clears from the same memref
                        bool clearsThisMemref = false;
                        clearLoop.walk([&](ClearKeyOp clearOp) {
                          Value clearedKey = clearOp.getRotKey();
                          while (auto useKey =
                                     clearedKey.getDefiningOp<UseKeyOp>()) {
                            clearedKey = useKey.getRotKey();
                          }
                          if (auto clearMemLoad =
                                  clearedKey.getDefiningOp<memref::LoadOp>()) {
                            if (clearMemLoad.getMemRef() == memref) {
                              clearsThisMemref = true;
                            }
                          }
                        });

                        if (clearsThisMemref) {
                          llvm::errs()
                              << "      Found existing memref.load of key "
                              << postloopKeyIndex << " before clear loop\n";
                          // Wrap the clear to skip this key index
                          clearLoop.walk([&](ClearKeyOp clearOp) {
                            Value clearedKey = clearOp.getRotKey();
                            while (auto useKey =
                                       clearedKey.getDefiningOp<UseKeyOp>()) {
                              clearedKey = useKey.getRotKey();
                            }
                            if (auto clearMemLoad =
                                    clearedKey
                                        .getDefiningOp<memref::LoadOp>()) {
                              if (clearMemLoad.getMemRef() == memref &&
                                  clearMemLoad.getIndices().size() == 1 &&
                                  clearMemLoad.getIndices()[0] ==
                                      clearLoop.getInductionVar()) {
                                wrapClearForPostloop(clearOp, loadedIdx,
                                                     clearLoop);
                              }
                            }
                          });

                          // Replace postloop load with use of the existing
                          // memref.load
                          replacePostloopLoadWithExistingMemrefLoad(loadOp,
                                                                    memLoad);
                          return;
                        }
                      }

                      if (!isa<LoadKeyOp, ClearKeyOp, UseKeyOp, AssumeLoadedOp>(
                              clearSearchOp)) {
                        clearDist++;
                      }
                      clearSearchOp = clearSearchOp->getNextNode();
                    }
                  }
                }
              }
            }
          }

          if (!isa<LoadKeyOp, ClearKeyOp, UseKeyOp, AssumeLoadedOp>(searchOp)) {
            distance++;
          }

          searchOp = searchOp->getPrevNode();
        }
      }  // End strategy 1

      // Strategy 2: Fallback to original loop-based pattern matching
      // Look for a loop immediately before the postloop load
      {
        Operation *searchOp = loadOp->getPrevNode();
        int distance = 0;

        while (searchOp && distance < 50) {
          if (auto affineLoop = dyn_cast<affine::AffineForOp>(searchOp)) {
            // Check if this loop has a pattern of:
            // %val = memref.load %alloca[%iv]
            // ...
            // kmrt.clear_key %val

            Value memrefAlloca = nullptr;
            ClearKeyOp clearToWrap = nullptr;

            affineLoop.walk([&](ClearKeyOp clearOp) {
              Value clearedKey = clearOp.getRotKey();

              // Trace back through any use_key operations to find the
              // memref.load
              while (auto useKeyOp = clearedKey.getDefiningOp<UseKeyOp>()) {
                clearedKey = useKeyOp.getRotKey();
              }

              if (auto memLoad = clearedKey.getDefiningOp<memref::LoadOp>()) {
                // Check if the load uses the loop IV
                Value loopIV = affineLoop.getInductionVar();
                if (memLoad.getIndices().size() == 1 &&
                    memLoad.getIndices()[0] == loopIV) {
                  // This clear uses a value from memref[iv]
                  memrefAlloca = memLoad.getMemRef();
                  clearToWrap = clearOp;
                  // llvm::errs() << "    Found postloop opportunity: loop
                  // clears "
                  //              << "memref[iv], postloop loads key "
                  //              << postloopKeyIndex << "\n";
                }
              }
            });

            if (clearToWrap && memrefAlloca) {
              // First, try to analyze what's actually in the memref
              std::optional<int64_t> memrefIndex =
                  findKeyInMemref(memrefAlloca, postloopKeyIndex);

              // If we can't prove what's in the memref, make an assumption for
              // epilogue loops
              if (!memrefIndex) {
                // For BSGS epilogue loops, assume memref[i] contains rotation
                // key i
                if (auto loopTypeAttr = affineLoop->getAttrOfType<StringAttr>(
                        "bsgs.loop_type")) {
                  if (loopTypeAttr.getValue() == "epilogue") {
                    // Check if the postloop key is within loop bounds
                    int64_t lb = affineLoop.getConstantLowerBound();
                    int64_t ub = affineLoop.getConstantUpperBound();
                    if (postloopKeyIndex >= lb && postloopKeyIndex < ub) {
                      memrefIndex = postloopKeyIndex;
                      llvm::errs()
                          << "      Assuming epilogue loop pattern: memref["
                          << postloopKeyIndex << "] contains key "
                          << postloopKeyIndex << "\n";
                    }
                  }
                }
              }

              if (memrefIndex) {
                // Perform the postloop merge
                wrapClearForPostloop(clearToWrap, *memrefIndex, affineLoop);
                replacePostloopLoadWithMemrefAccess(loadOp, memrefAlloca,
                                                    *memrefIndex);
                return;
              }
            }
          }

          if (!isa<LoadKeyOp, ClearKeyOp, UseKeyOp, AssumeLoadedOp>(searchOp)) {
            distance++;
          }

          searchOp = searchOp->getPrevNode();
        }
      }  // End strategy 2
    });
  }

  // Wrap a ClearKeyOp with affine.if to skip a specific key index
  void wrapClearForPostloop(ClearKeyOp clearOp, int64_t keyIndexToSkip,
                            affine::AffineForOp loop) {
    OpBuilder builder(clearOp);
    Location loc = clearOp.getLoc();
    Value loopIV = loop.getInductionVar();

    // Create affine.if: if (iv == keyIndexToSkip) { skip } else { clear }
    auto dimExpr = builder.getAffineDimExpr(0);
    auto condExpr = dimExpr - keyIndexToSkip;
    auto condSet =
        IntegerSet::get(1, 0, {condExpr}, {true});  // true = equality

    auto ifOp = builder.create<affine::AffineIfOp>(loc, TypeRange{}, condSet,
                                                   ValueRange{loopIV},
                                                   /*withElseRegion=*/true);

    // Then branch: skip (empty)
    // (already has implicit terminator)

    // Else branch: do the clear
    {
      OpBuilder elseBuilder(ifOp.getElseBlock(), ifOp.getElseBlock()->begin());
      elseBuilder.clone(*clearOp);
    }

    // Erase the original clear
    clearOp.erase();

    llvm::errs() << "      Wrapped postloop clear with affine.if to skip key "
                 << keyIndexToSkip << "\n";
  }

  // Replace a postloop LoadKeyOp with an existing memref.load
  void replacePostloopLoadWithExistingMemrefLoad(
      LoadKeyOp loadOp, memref::LoadOp existingMemLoad) {
    OpBuilder builder(loadOp);
    Location loc = loadOp.getLoc();

    // Create use_key from the existing memref.load
    auto useKeyOp = builder.create<UseKeyOp>(loc, loadOp.getRotKey().getType(),
                                             existingMemLoad.getResult());

    // Replace the original load
    loadOp.getRotKey().replaceAllUsesWith(useKeyOp.getRotKey());
    loadOp.erase();

    llvm::errs() << "      Replaced postloop load with existing memref.load\n";
  }

  // Replace a postloop LoadKeyOp with memref access
  void replacePostloopLoadWithMemrefAccess(LoadKeyOp loadOp, Value memrefAlloca,
                                           int64_t keyIndex) {
    OpBuilder builder(loadOp);
    Location loc = loadOp.getLoc();

    // Create constant for the index
    Value indexVal = builder.create<arith::ConstantIndexOp>(loc, keyIndex);

    // Load from memref
    Value loadedKey =
        builder.create<memref::LoadOp>(loc, memrefAlloca, ValueRange{indexVal});

    // Create use_key
    auto useKeyOp =
        builder.create<UseKeyOp>(loc, loadOp.getRotKey().getType(), loadedKey);

    // Replace the original load
    loadOp.getRotKey().replaceAllUsesWith(useKeyOp.getRotKey());
    loadOp.erase();

    llvm::errs() << "      Replaced postloop load with memref access at index "
                 << keyIndex << "\n";
  }

  void extendAffineIfForPreloopKeys(
      affine::AffineForOp loop,
      SmallVector<std::tuple<int64_t, LoadKeyOp, ClearKeyOp>> &preloopKeys,
      DataFlowSolver &solver) {
    llvm::errs() << "    extendAffineIfForPreloopKeys: processing loop\n";

    // Find existing affine.if operations created by BSGS pass
    // These check outer_iv == 0 and load/use keys
    SmallVector<affine::AffineIfOp> affineIfsToExtend;

    loop.walk([&](affine::AffineIfOp ifOp) {
      // Check if this is a load guard (returns a rotation key)
      if (ifOp.getNumResults() > 0 &&
          isa<RotKeyType>(ifOp.getResult(0).getType())) {
        affineIfsToExtend.push_back(ifOp);
      }
    });

    llvm::errs() << "    Found " << affineIfsToExtend.size()
                 << " affine.if operations to potentially extend\n";

    for (auto ifOp : affineIfsToExtend) {
      // For each preloop key, check if it should be merged here
      for (auto &[preloopKeyIndex, preloadOp, clearOp] : preloopKeys) {
        // Use affine set analysis to find merge conditions
        auto mergeOpp = kmrt::analyzeMergeOpportunity(preloopKeyIndex, loop);

        if (!mergeOpp || !mergeOpp->isValid()) continue;

        llvm::errs() << "      Extending affine.if for preloop key "
                     << preloopKeyIndex << "\n";

        // Get the merge conditions (which IVs must equal what values)
        // For now, handle simple case: one condition
        if (mergeOpp->mergeConditions.empty()) continue;

        auto [mergeIV, mergeValue] = mergeOpp->mergeConditions[0];

        // Extend the affine.if to add: outer_iv == 0 && inner_iv == mergeValue
        extendAffineIfCondition(ifOp, mergeIV, mergeValue, preloadOp);

        // Remove the preloop clear operation
        if (clearOp && clearOp.getOperation()->getBlock()) {
          llvm::errs() << "      Removing preloop clear for key "
                       << preloopKeyIndex << "\n";
          clearOp.erase();
        }
      }
    }
  }

  // Extend an existing affine.if to add a new condition branch
  void extendAffineIfCondition(affine::AffineIfOp ifOp, Value newCondIV,
                               int64_t newCondValue, LoadKeyOp preloadOp) {
    OpBuilder builder(ifOp);
    Location loc = ifOp.getLoc();

    // Create new affine.if with extended condition:
    // if (outer_iv == 0 && inner_iv == K) { use_preloaded }
    // else { original_if }

    // Build condition: inner_iv == K
    auto dimExpr = builder.getAffineDimExpr(0);
    auto condExpr = dimExpr - newCondValue;
    auto condSet = IntegerSet::get(1, 0, {condExpr}, {true});

    auto newIfOp = builder.create<affine::AffineIfOp>(
        loc, ifOp.getResultTypes(), condSet, ValueRange{newCondIV},
        /*withElseRegion=*/true);

    // Then branch: use preloaded key
    {
      OpBuilder thenBuilder(newIfOp.getThenBlock(),
                            newIfOp.getThenBlock()->begin());
      // Use the result type from the original affine.if to ensure type
      // compatibility
      auto useKeyOp = thenBuilder.create<UseKeyOp>(
          loc, ifOp.getResultTypes()[0], preloadOp.getRotKey());
      thenBuilder.create<affine::AffineYieldOp>(loc, useKeyOp.getRotKey());
    }

    // Else branch: clone original affine.if
    {
      OpBuilder elseBuilder(newIfOp.getElseBlock(),
                            newIfOp.getElseBlock()->begin());
      IRMapping mapping;
      auto clonedIf =
          cast<affine::AffineIfOp>(elseBuilder.clone(*ifOp, mapping));
      elseBuilder.create<affine::AffineYieldOp>(loc, clonedIf.getResult(0));
    }

    // Replace uses and erase original
    ifOp.getResult(0).replaceAllUsesWith(newIfOp.getResult(0));
    ifOp.erase();

    llvm::errs() << "      Successfully extended affine.if condition\n";
  }

  // Create standalone merge for loops without existing affine.if guards
  void createStandaloneMerge(
      affine::AffineForOp loop,
      SmallVector<std::tuple<int64_t, LoadKeyOp, ClearKeyOp>> &preloopKeys,
      DataFlowSolver &solver) {
    llvm::errs()
        << "    createStandaloneMerge: checking for standalone opportunities\n";

    // Find LoadKeyOp operations in the loop and collect all preloop keys for
    // each
    DenseMap<LoadKeyOp, SmallVector<int64_t>> loadToPreloopKeys;

    loop.walk([&](LoadKeyOp loadOp) {
      // Check if this load's index value uses the loop IV
      Value indexValue = loadOp.getIndex();

      // Check if it's a direct use of IV (with or without index_cast)
      Value loopIV = loop.getInductionVar();

      bool usesLoopIV = false;

      // Direct use of loop IV
      if (indexValue == loopIV) {
        usesLoopIV = true;
      }
      // Try to trace through index_cast
      else if (auto castOp = indexValue.getDefiningOp<arith::IndexCastOp>()) {
        if (castOp.getIn() == loopIV) {
          usesLoopIV = true;
        }
      }

      if (usesLoopIV) {
        // This load uses the loop IV directly - collect ALL preloop keys
        for (auto &[preloopKeyIndex, preloadOp, clearOp] : preloopKeys) {
          loadToPreloopKeys[loadOp].push_back(preloopKeyIndex);
          llvm::errs() << "      Found load to wrap for key " << preloopKeyIndex
                       << "\n";
        }
      }
    });

    // Wrap each LoadKeyOp once with all its preloop keys
    SmallVector<std::pair<Value, int64_t>> wrappedResults;
    for (auto &[loadOp, keyIndices] : loadToPreloopKeys) {
      Value newResult = wrapLoadWithAffineIfMultipleKeys(loadOp, keyIndices,
                                                         loop, preloopKeys);
      if (newResult) {
        for (int64_t keyIndex : keyIndices) {
          wrappedResults.push_back({newResult, keyIndex});
        }
      }
    }

    // Wrap ClearKeyOp operations
    SmallVector<std::pair<ClearKeyOp, SmallVector<int64_t>>> clearsToWrap;
    loop.walk([&](ClearKeyOp clearOp) {
      Value key = clearOp.getRotKey();

      // Check if this clear corresponds to a wrapped load's new result
      for (auto &[wrappedResult, preloopKeyIndex] : wrappedResults) {
        if (key == wrappedResult) {
          SmallVector<int64_t> keysToSkip;
          keysToSkip.push_back(preloopKeyIndex);
          clearsToWrap.push_back({clearOp, keysToSkip});
          llvm::errs() << "      Found clear to wrap, skipping key "
                       << preloopKeyIndex << "\n";
          break;
        }
      }
    });

    // Wrap each ClearKeyOp with affine.if
    for (auto &[clearOp, keysToSkip] : clearsToWrap) {
      wrapClearWithAffineIf(clearOp, keysToSkip, loop);
    }

    // Remove preloop clears
    for (auto &[preloopKeyIndex, preloadOp, clearOp] : preloopKeys) {
      if (clearOp && clearOp.getOperation()->getBlock()) {
        llvm::errs() << "      Removing preloop clear for key "
                     << preloopKeyIndex << "\n";
        clearOp.erase();
      }
    }
  }

  // Wrap a LoadKeyOp with affine.if for preloop merge (single key)
  Value wrapLoadWithAffineIf(
      LoadKeyOp loadOp, int64_t preloopKeyIndex, affine::AffineForOp loop,
      SmallVector<std::tuple<int64_t, LoadKeyOp, ClearKeyOp>> &preloopKeys) {
    OpBuilder builder(loadOp);
    Location loc = loadOp.getLoc();
    Value loopIV = loop.getInductionVar();

    // Find the preload operation
    LoadKeyOp preloadOp = nullptr;
    for (auto &[keyIndex, preload, clear] : preloopKeys) {
      if (keyIndex == preloopKeyIndex) {
        preloadOp = preload;
        break;
      }
    }
    if (!preloadOp) return nullptr;

    // Create affine.if: if (iv == preloopKeyIndex) { use preloaded } else {
    // load }
    auto dimExpr = builder.getAffineDimExpr(0);
    auto condExpr = dimExpr - preloopKeyIndex;
    auto condSet = IntegerSet::get(1, 0, {condExpr}, {true});

    auto ifOp = builder.create<affine::AffineIfOp>(
        loc, loadOp.getRotKey().getType(), condSet, ValueRange{loopIV},
        /*withElseRegion=*/true);

    // Then branch: use preloaded key
    {
      OpBuilder thenBuilder(ifOp.getThenBlock(), ifOp.getThenBlock()->begin());
      auto useKeyOp = thenBuilder.create<UseKeyOp>(
          loc, loadOp.getRotKey().getType(), preloadOp.getRotKey());
      thenBuilder.create<affine::AffineYieldOp>(loc, useKeyOp.getRotKey());
    }

    // Else branch: clone original load
    {
      OpBuilder elseBuilder(ifOp.getElseBlock(), ifOp.getElseBlock()->begin());
      IRMapping mapping;
      auto clonedLoad = cast<LoadKeyOp>(elseBuilder.clone(*loadOp, mapping));
      elseBuilder.create<affine::AffineYieldOp>(loc, clonedLoad.getRotKey());
    }

    // Replace uses and erase original
    loadOp.getRotKey().replaceAllUsesWith(ifOp.getResult(0));
    loadOp.erase();

    llvm::errs() << "      Wrapped LoadKeyOp with affine.if for key "
                 << preloopKeyIndex << "\n";

    return ifOp.getResult(0);
  }

  // Wrap a LoadKeyOp with nested affine.if for multiple preloop keys
  Value wrapLoadWithAffineIfMultipleKeys(
      LoadKeyOp loadOp, SmallVector<int64_t> &preloopKeyIndices,
      affine::AffineForOp loop,
      SmallVector<std::tuple<int64_t, LoadKeyOp, ClearKeyOp>> &preloopKeys) {
    // If only one key, use the simpler single-key version
    if (preloopKeyIndices.size() == 1) {
      return wrapLoadWithAffineIf(loadOp, preloopKeyIndices[0], loop,
                                  preloopKeys);
    }

    OpBuilder builder(loadOp);
    Location loc = loadOp.getLoc();
    Value loopIV = loop.getInductionVar();

    // Build nested affine.if structure:
    // if (iv == key[0]) { use key[0] }
    // else { if (iv == key[1]) { use key[1] }
    //        else { ... if (iv == key[n]) { use key[n] }
    //               else { original load } } }

    // Helper to recursively build nested ifs
    std::function<Value(size_t)> buildNestedIf = [&](size_t keyIdx) -> Value {
      if (keyIdx >= preloopKeyIndices.size()) {
        // Base case: clone the original load
        IRMapping mapping;
        auto clonedLoad = cast<LoadKeyOp>(builder.clone(*loadOp, mapping));
        return clonedLoad.getRotKey();
      }

      int64_t preloopKeyIndex = preloopKeyIndices[keyIdx];

      // Find the preload operation for this key
      LoadKeyOp preloadOp = nullptr;
      for (auto &[keyIndex, preload, clear] : preloopKeys) {
        if (keyIndex == preloopKeyIndex) {
          preloadOp = preload;
          break;
        }
      }
      if (!preloadOp) return buildNestedIf(keyIdx + 1);

      // Create affine.if for this key
      auto dimExpr = builder.getAffineDimExpr(0);
      auto condExpr = dimExpr - preloopKeyIndex;
      auto condSet = IntegerSet::get(1, 0, {condExpr}, {true});

      auto ifOp = builder.create<affine::AffineIfOp>(
          loc, loadOp.getRotKey().getType(), condSet, ValueRange{loopIV},
          /*withElseRegion=*/true);

      // Mark nested affine.if operations (not the outermost one) with nesting
      // level so the emitter can generate unique variable names with suffix
      if (keyIdx > 0) {
        ifOp->setAttr("nesting_level", builder.getI64IntegerAttr(keyIdx));
      }

      // Then branch: use preloaded key
      {
        OpBuilder thenBuilder(ifOp.getThenBlock(),
                              ifOp.getThenBlock()->begin());
        auto useKeyOp = thenBuilder.create<UseKeyOp>(
            loc, loadOp.getRotKey().getType(), preloadOp.getRotKey());
        thenBuilder.create<affine::AffineYieldOp>(loc, useKeyOp.getRotKey());
      }

      // Else branch: recursively build the next nested if or the load
      {
        OpBuilder elseBuilder(ifOp.getElseBlock(),
                              ifOp.getElseBlock()->begin());
        builder.setInsertionPointToStart(ifOp.getElseBlock());
        Value elseResult = buildNestedIf(keyIdx + 1);
        builder.setInsertionPointToEnd(ifOp.getElseBlock());
        builder.create<affine::AffineYieldOp>(loc, elseResult);
      }

      llvm::errs() << "      Wrapped LoadKeyOp with affine.if for key "
                   << preloopKeyIndex << "\n";

      return ifOp.getResult(0);
    };

    // Build the nested structure starting from the first key
    builder.setInsertionPoint(loadOp);
    Value result = buildNestedIf(0);

    // Replace uses and erase original
    loadOp.getRotKey().replaceAllUsesWith(result);
    loadOp.erase();

    return result;
  }

  // Wrap a ClearKeyOp with affine.if to skip certain keys
  void wrapClearWithAffineIf(ClearKeyOp clearOp,
                             SmallVector<int64_t> &keysToSkip,
                             affine::AffineForOp loop) {
    OpBuilder builder(clearOp);
    Location loc = clearOp.getLoc();
    Value loopIV = loop.getInductionVar();

    // Create nested affine.if for each key to skip
    // Structure: if (iv == key1) {} else { if (iv == key2) {} else { clear } }

    Operation *lastOp = clearOp;
    for (int64_t keyIndex : keysToSkip) {
      auto dimExpr = builder.getAffineDimExpr(0);
      auto condExpr = dimExpr - keyIndex;
      auto condSet = IntegerSet::get(1, 0, {condExpr}, {true});

      builder.setInsertionPoint(lastOp);
      auto ifOp = builder.create<affine::AffineIfOp>(loc, TypeRange{}, condSet,
                                                     ValueRange{loopIV},
                                                     /*withElseRegion=*/true);

      // Then branch: empty (skip clear)
      // Else branch will be filled in next iteration or with the clear

      lastOp = ifOp;
    }

    // Move the clear into the innermost else block before the terminator
    if (auto ifOp = dyn_cast<affine::AffineIfOp>(lastOp)) {
      Block *elseBlock = ifOp.getElseBlock();
      // If the else block has a terminator, insert before it
      if (!elseBlock->empty() &&
          elseBlock->back().hasTrait<OpTrait::IsTerminator>()) {
        clearOp->moveBefore(&elseBlock->back());
      } else {
        clearOp->moveBefore(elseBlock, elseBlock->end());
      }
      llvm::errs() << "      Wrapped ClearKeyOp with affine.if\n";
    }
  }

  // Peel key management (loads/clears) out of inner loops for better overlap
  void peelInnerLoopKeyManagement(affine::AffineForOp outerLoop) {
    // llvm::errs() << "  Analyzing outer loop for peeling opportunities\n";

    // Find the inner loop
    affine::AffineForOp innerLoop = nullptr;
    outerLoop.walk([&](affine::AffineForOp loop) {
      if (loop != outerLoop && !innerLoop) {
        innerLoop = loop;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (!innerLoop) {
      llvm::errs() << "    No inner loop found\n";
      return;
    }

    // llvm::errs() << "    Found inner loop, checking for load/clear
    // patterns\n";

    // Find affine.if operations for loads and clears in the inner loop
    affine::AffineIfOp loadIf = nullptr;
    affine::AffineIfOp clearIf = nullptr;
    Value memrefAlloca = nullptr;

    innerLoop.walk([&](affine::AffineIfOp ifOp) {
      // Check if this is a load guard (returns a rotation key)
      if (ifOp.getNumResults() > 0 &&
          isa<RotKeyType>(ifOp.getResult(0).getType())) {
        // Check if the then block contains a LoadKeyOp
        bool hasLoad = false;
        ifOp.getThenBlock()->walk([&](LoadKeyOp) {
          hasLoad = true;
          return WalkResult::interrupt();
        });
        if (hasLoad) {
          loadIf = ifOp;
          // Find memref.store to identify the allocation
          ifOp.getThenBlock()->walk([&](memref::StoreOp storeOp) {
            memrefAlloca = storeOp.getMemRef();
            return WalkResult::interrupt();
          });
        }
      } else if (ifOp.getNumResults() == 0) {
        // Check if this is a clear guard
        bool hasClear = false;
        ifOp.getThenBlock()->walk([&](ClearKeyOp) {
          hasClear = true;
          return WalkResult::interrupt();
        });
        if (hasClear) {
          clearIf = ifOp;
        }
      }
    });

    if (!loadIf || !clearIf || !memrefAlloca) {
      // llvm::errs() << "    No load/clear pattern found suitable for
      // peeling\n";
      return;
    }

    // llvm::errs() << "    Found load/clear pattern, peeling key management\n";

    OpBuilder builder(innerLoop);
    Location loc = innerLoop.getLoc();

    // Extract the condition from loadIf (e.g., outer_iv == 0)
    IntegerSet loadCondSet = loadIf.getIntegerSet();
    ValueRange loadCondOperands = loadIf.getOperands();

    // Extract the condition from clearIf (e.g., outer_iv == last)
    IntegerSet clearCondSet = clearIf.getIntegerSet();
    ValueRange clearCondOperands = clearIf.getOperands();

    // Get loop bounds
    int64_t lowerBound = innerLoop.getConstantLowerBound();
    int64_t upperBound = innerLoop.getConstantUpperBound();

    // Create load loop before inner loop
    auto loadGuard = builder.create<affine::AffineIfOp>(
        loc, TypeRange{}, loadCondSet, loadCondOperands,
        /*withElseRegion=*/false);

    OpBuilder loadBuilder(loadGuard.getThenBlock(),
                          loadGuard.getThenBlock()->begin());
    auto loadLoop =
        loadBuilder.create<affine::AffineForOp>(loc, lowerBound, upperBound);

    // Build load loop body
    OpBuilder loadBodyBuilder(loadLoop.getBody(), loadLoop.getBody()->begin());
    Value loadLoopIV = loadLoop.getInductionVar();

    // Clone the load operations from the then block
    IRMapping loadMapping;
    loadMapping.map(innerLoop.getInductionVar(), loadLoopIV);

    for (auto &op : *loadIf.getThenBlock()) {
      if (!isa<affine::AffineYieldOp>(op)) {
        loadBodyBuilder.clone(op, loadMapping);
      }
    }

    // Modify inner loop to only use keys (replace loadIf with just memref.load
    // + use_key)
    builder.setInsertionPoint(loadIf);
    Value innerLoopIV = innerLoop.getInductionVar();
    auto memrefLoad = builder.create<memref::LoadOp>(loc, memrefAlloca,
                                                     ValueRange{innerLoopIV});
    auto useKey = builder.create<UseKeyOp>(loc, loadIf.getResult(0).getType(),
                                           memrefLoad.getResult());

    loadIf.getResult(0).replaceAllUsesWith(useKey.getResult());
    loadIf.erase();

    // Create clear loop after inner loop
    builder.setInsertionPointAfter(innerLoop);
    auto clearGuard = builder.create<affine::AffineIfOp>(
        loc, TypeRange{}, clearCondSet, clearCondOperands,
        /*withElseRegion=*/false);

    OpBuilder clearBuilder(clearGuard.getThenBlock(),
                           clearGuard.getThenBlock()->begin());
    auto clearLoop =
        clearBuilder.create<affine::AffineForOp>(loc, lowerBound, upperBound);

    // Build clear loop body
    OpBuilder clearBodyBuilder(clearLoop.getBody(),
                               clearLoop.getBody()->begin());
    Value clearLoopIV = clearLoop.getInductionVar();

    // Clone the clear operations from the then block
    IRMapping clearMapping;
    clearMapping.map(innerLoop.getInductionVar(), clearLoopIV);

    for (auto &op : *clearIf.getThenBlock()) {
      if (!isa<affine::AffineYieldOp>(op)) {
        clearBodyBuilder.clone(op, clearMapping);
      }
    }

    // Remove the clear guard from inner loop
    clearIf.erase();

    llvm::errs() << "    Successfully peeled load and clear loops\n";
  }
};

}  // namespace kmrt
}  // namespace heir
}  // namespace mlir
