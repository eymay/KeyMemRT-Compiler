#include "lib/Dialect/KMRT/Transforms/KeyPrefetching.h"

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <vector>

#include "lib/Dialect/KMRT/IR/KMRTOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project

#define DEBUG_TYPE "kmrt-key-prefetching"

namespace mlir {
namespace heir {
namespace kmrt {

#define GEN_PASS_DEF_KMRTKEYPREFETCHING
#include "lib/Dialect/KMRT/Transforms/Passes.h.inc"

namespace {

// Abstract value representing a key index
struct AbstractKeyIndex {
  // Constant value if known, otherwise nullopt
  std::optional<int64_t> constantValue;

  // Affine expression if it's a function of loop IVs
  std::optional<AffineExpr> affineExpr;

  // Mapping from dimension/symbol positions to Values (for affine expressions)
  SmallVector<Value> affineOperands;

  // Is this a known constant?
  bool isConstant() const { return constantValue.has_value(); }

  // Is this an affine expression?
  bool isAffine() const { return affineExpr.has_value(); }

  // Try to evaluate to a constant given a map of induction variables to their
  // values
  std::optional<int64_t> tryEvaluate(
      const DenseMap<Value, int64_t>& ivValues) const {
    if (isConstant()) {
      return constantValue;
    }

    if (isAffine() && affineExpr.has_value()) {
      auto expr = *affineExpr;

      // Count how many dimensions and symbols the expression expects
      unsigned numDims = 0;
      unsigned numSyms = 0;
      expr.walk([&](AffineExpr subExpr) {
        if (auto dimExpr = dyn_cast<AffineDimExpr>(subExpr)) {
          numDims = std::max(numDims, dimExpr.getPosition() + 1);
        } else if (auto symExpr = dyn_cast<AffineSymbolExpr>(subExpr)) {
          numSyms = std::max(numSyms, symExpr.getPosition() + 1);
        }
      });

      // Operands are ordered: first numDims dimensions, then numSyms symbols
      if (affineOperands.size() != numDims + numSyms) {
        return std::nullopt;
      }

      // Extract dimension and symbol values
      SmallVector<int64_t> dimValues;
      SmallVector<int64_t> symValues;

      for (unsigned i = 0; i < numDims; i++) {
        auto it = ivValues.find(affineOperands[i]);
        if (it == ivValues.end()) {
          return std::nullopt;
        }
        dimValues.push_back(it->second);
      }

      for (unsigned i = 0; i < numSyms; i++) {
        auto it = ivValues.find(affineOperands[numDims + i]);
        if (it == ivValues.end()) {
          return std::nullopt;
        }
        symValues.push_back(it->second);
      }

      // Replace dimensions and symbols with concrete values
      SmallVector<AffineExpr> dimReplacements;
      for (auto val : dimValues) {
        dimReplacements.push_back(
            getAffineConstantExpr(val, expr.getContext()));
      }
      SmallVector<AffineExpr> symReplacements;
      for (auto val : symValues) {
        symReplacements.push_back(
            getAffineConstantExpr(val, expr.getContext()));
      }

      auto constantExpr =
          expr.replaceDimsAndSymbols(dimReplacements, symReplacements);
      if (auto constExpr = dyn_cast<AffineConstantExpr>(constantExpr)) {
        return constExpr.getValue();
      }
    }

    return std::nullopt;
  }

  // Check if two abstract indices could be equal
  bool mightEqual(const AbstractKeyIndex& other,
                  const DenseMap<Value, int64_t>& ivValues) const {
    auto thisVal = tryEvaluate(ivValues);
    auto otherVal = other.tryEvaluate(ivValues);

    if (thisVal && otherVal) {
      return *thisVal == *otherVal;
    }

    // If we can't evaluate, conservatively assume they might be equal
    // if they have the same structure
    if (isConstant() && other.isConstant()) {
      return constantValue == other.constantValue;
    }

    if (isAffine() && other.isAffine() && affineExpr && other.affineExpr) {
      // Check if expressions are structurally equal
      if (*affineExpr == *other.affineExpr &&
          affineOperands.size() == other.affineOperands.size()) {
        for (size_t i = 0; i < affineOperands.size(); i++) {
          if (affineOperands[i] != other.affineOperands[i]) {
            return false;
          }
        }
        return true;
      }
    }

    return false;
  }
};

// State tracker for verification
struct VerificationState {
  // Set of prefetched key indices at each program point
  std::set<int64_t> prefetchedConstants;

  // Map from abstract indices to whether they're prefetched
  SmallVector<AbstractKeyIndex> prefetchedIndices;

  // Check if a key index has been prefetched
  bool isPrefetched(const AbstractKeyIndex& idx,
                    const DenseMap<Value, int64_t>& ivValues) const {
    // Check constants first
    if (idx.isConstant()) {
      return prefetchedConstants.count(*idx.constantValue) > 0;
    }

    // Check if any prefetched index matches
    for (const auto& prefetched : prefetchedIndices) {
      if (idx.mightEqual(prefetched, ivValues)) {
        return true;
      }
    }

    return false;
  }

  // Mark a key index as prefetched
  void markPrefetched(const AbstractKeyIndex& idx) {
    if (idx.isConstant()) {
      prefetchedConstants.insert(*idx.constantValue);
    }
    prefetchedIndices.push_back(idx);
  }

  // Merge with another state (for control flow joins)
  void merge(const VerificationState& other) {
    // Only keep constants that are in both
    std::set<int64_t> intersection;
    std::set_intersection(
        prefetchedConstants.begin(), prefetchedConstants.end(),
        other.prefetchedConstants.begin(), other.prefetchedConstants.end(),
        std::inserter(intersection, intersection.begin()));
    prefetchedConstants = intersection;

    // For affine indices, keep ones that match in both
    SmallVector<AbstractKeyIndex> mergedIndices;
    DenseMap<Value, int64_t> emptyIvValues;
    for (const auto& idx : prefetchedIndices) {
      for (const auto& otherIdx : other.prefetchedIndices) {
        if (idx.mightEqual(otherIdx, emptyIvValues)) {
          mergedIndices.push_back(idx);
          break;
        }
      }
    }
    prefetchedIndices = mergedIndices;
  }
};

// Operation cost table - relative timing units
inline int64_t getOperationCost(Operation* op) {
  return llvm::TypeSwitch<Operation*, int64_t>(op)
      // Computational operations
      .Case<openfhe::AddOp, openfhe::SubOp, openfhe::AddPlainOp,
            openfhe::SubPlainOp>([](auto) { return 1; })
      .Case<openfhe::MulOp, openfhe::MulPlainOp, openfhe::MulNoRelinOp,
            openfhe::MulConstOp>([](auto) { return 10; })
      .Case<openfhe::ChebyshevOp>([](auto) { return 50; })
      .Case<openfhe::BootstrapOp>([](auto) { return 100; })
      .Case<openfhe::RotOp, openfhe::AutomorphOp, openfhe::FastRotationOp,
            RotationOp>([](auto) { return 15; })
      .Case<openfhe::FastRotationPrecomputeOp>([](auto) { return 5; })
      .Case<openfhe::RelinOp>([](auto) { return 8; })
      .Case<openfhe::ModReduceOp, openfhe::LevelReduceOp>(
          [](auto) { return 3; })
      .Case<openfhe::KeySwitchOp>([](auto) { return 12; })
      // Key management operations (0 cost for prefetch calculation)
      .Case<LoadKeyOp, ClearKeyOp, PrefetchKeyOp>([](auto) { return 0; })
      // Default case
      .Default([](Operation*) { return 1; });
}

struct KMRTKeyPrefetching : impl::KMRTKeyPrefetchingBase<KMRTKeyPrefetching> {
  using impl::KMRTKeyPrefetchingBase<
      KMRTKeyPrefetching>::KMRTKeyPrefetchingBase;

  void runOnOperation() override {
    Operation* op = getOperation();

    // Initialize dominance info
    DominanceInfo domInfoObj(op);
    domInfo = &domInfoObj;

    // If runtime-delegated mode is enabled, collect all keys and emit them at
    // the beginning
    if (runtimeDelegated.getValue()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "KMRTKeyPrefetching: Runtime delegated mode enabled\n");
      createRuntimeDelegatedPrefetchList(op);
      return;
    }

    // If skip-insertion is enabled, only run verification
    if (!skipInsertion.getValue()) {
      // Find all outermost affine loops and process them
      // This ensures we handle nested loops together
      SmallVector<affine::AffineForOp> outermostLoops;
      op->walk([&](affine::AffineForOp forOp) {
        // Only process if not nested inside another affine.for
        if (!forOp->getParentOfType<affine::AffineForOp>()) {
          outermostLoops.push_back(forOp);
        }
      });

      LLVM_DEBUG(llvm::dbgs() << "KMRTKeyPrefetching: Found "
                              << outermostLoops.size() << " outermost loops\n");

      // Process each outermost loop
      for (auto forOp : outermostLoops) {
        processAffineLoop(forOp);
      }

      // Process non-loop load_key operations
      SmallVector<LoadKeyOp> nonLoopLoadOps;
      op->walk([&](LoadKeyOp loadOp) {
        if (!loadOp->getParentOfType<affine::AffineForOp>()) {
          nonLoopLoadOps.push_back(loadOp);
        }
      });

      // Try to batch post-loop keys that follow a loop
      batchPostLoopPrefetches(nonLoopLoadOps);

      // Process remaining non-loop loads individually
      for (auto loadOp : nonLoopLoadOps) {
        // Skip if already processed by batching
        if (!loadOp->hasAttr("kmrt.prefetch_batched")) {
          insertPrefetchForLoad(loadOp);
        }
      }
    } else {
      LLVM_DEBUG(
          llvm::dbgs()
          << "KMRTKeyPrefetching: Skipping insertion (skip-insertion=true)\n");
    }

    // Verify prefetches if requested
    if (verifyPrefetches.getValue()) {
      LLVM_DEBUG(llvm::dbgs() << "KMRTKeyPrefetching: Running verification\n");
      if (failed(verifyAllPrefetches(op))) {
        // Don't emit module-level error - errors are already emitted at
        // load_key operations
        signalPassFailure();
      }
    }
  }

 private:
  DominanceInfo* domInfo = nullptr;

  // Batch post-loop prefetches to spread load on deserializing thread
  void batchPostLoopPrefetches(SmallVectorImpl<LoadKeyOp>& nonLoopLoadOps) {
    // Group consecutive load_key operations and find their preceding loop
    SmallVector<SmallVector<LoadKeyOp>> loadGroups;
    SmallVector<affine::AffineForOp> precedingLoops;

    for (size_t i = 0; i < nonLoopLoadOps.size();) {
      LoadKeyOp firstLoad = nonLoopLoadOps[i];

      // Find the immediately preceding loop
      affine::AffineForOp precedingLoop = findPrecedingLoop(firstLoad);

      if (!precedingLoop) {
        i++;
        continue;
      }

      // Collect consecutive loads that share the same preceding loop
      SmallVector<LoadKeyOp> group;
      group.push_back(firstLoad);

      size_t j = i + 1;
      while (j < nonLoopLoadOps.size()) {
        LoadKeyOp nextLoad = nonLoopLoadOps[j];
        affine::AffineForOp nextPrecedingLoop = findPrecedingLoop(nextLoad);

        if (nextPrecedingLoop != precedingLoop) {
          break;
        }

        // Check if this load is close enough to be in the same batch
        Operation* prevOp = nextLoad.getOperation();
        bool closeEnough = true;
        int distance = 0;
        while (prevOp && distance < 20) {
          if (prevOp == group.back().getOperation()) {
            break;
          }
          prevOp = prevOp->getPrevNode();
          distance++;
        }

        if (!closeEnough || distance >= 20) {
          break;
        }

        group.push_back(nextLoad);
        j++;
      }

      // Only batch if we have multiple keys or if the loop has enough
      // iterations
      if (group.size() >= 2 && precedingLoop.hasConstantBounds()) {
        int64_t lb = precedingLoop.getConstantLowerBound();
        int64_t ub = precedingLoop.getConstantUpperBound();
        int64_t step = precedingLoop.getStep().getSExtValue();
        int64_t numIters = (ub - lb) / step;

        // Only batch if loop has enough iterations to spread the load
        if (numIters >= 2) {
          loadGroups.push_back(group);
          precedingLoops.push_back(precedingLoop);

          LLVM_DEBUG(llvm::dbgs()
                     << "Batching " << group.size() << " post-loop keys across "
                     << numIters << " iterations\n");
        }
      }

      i = j;
    }

    // Process each batch
    for (size_t i = 0; i < loadGroups.size(); i++) {
      insertBatchedPrefetches(precedingLoops[i], loadGroups[i]);
    }
  }

  // Find the loop immediately before this load operation
  affine::AffineForOp findPrecedingLoop(LoadKeyOp loadOp) {
    Operation* current = loadOp.getOperation();

    // Walk backwards looking for a loop
    while (current) {
      current = current->getPrevNode();
      if (!current) {
        // Reached beginning of block, check parent
        Block* parentBlock = loadOp->getBlock();
        if (parentBlock && parentBlock->getParentOp()) {
          current = parentBlock->getParentOp();
          if (auto forOp = dyn_cast<affine::AffineForOp>(current)) {
            return forOp;
          }
          current = current->getPrevNode();
        }
      }

      if (auto forOp = dyn_cast_or_null<affine::AffineForOp>(current)) {
        return forOp;
      }

      // Skip operations that commonly appear between loop and post-loop loads
      if (current) {
        // Skip constants, loads, prefetches, and cleanup operations
        if (isa<arith::ConstantOp, arith::ConstantIndexOp, LoadKeyOp,
                PrefetchKeyOp, UseKeyOp>(current)) {
          continue;
        }

        // Skip OpenFHE operations (clears, arithmetic, rotations)
        if (current->getDialect() &&
            current->getDialect()->getNamespace() == "openfhe") {
          continue;
        }

        // Skip other common operations (tensor, memref)
        if (isa<tensor::ExtractSliceOp, memref::LoadOp, memref::StoreOp>(
                current)) {
          continue;
        }

        // Stop at control flow boundaries (other loops, conditionals, function
        // boundaries)
        if (isa<affine::AffineForOp, affine::AffineIfOp, scf::ForOp, scf::IfOp>(
                current) ||
            isa<func::FuncOp, func::ReturnOp>(current)) {
          break;
        }
      }
    }

    return nullptr;
  }

  // Insert batched prefetches for a group of post-loop keys
  void insertBatchedPrefetches(affine::AffineForOp forOp,
                               SmallVectorImpl<LoadKeyOp>& loadOps) {
    if (loadOps.empty()) return;

    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();

    int64_t numKeys = loadOps.size();
    LLVM_DEBUG(llvm::dbgs()
               << "Inserting batched prefetches for " << numKeys << " keys\n");

    // Create a memref to store the key indices
    auto indexType = builder.getIndexType();
    auto memrefType = MemRefType::get({numKeys}, indexType);
    Value keyIndicesMemref = builder.create<memref::AllocaOp>(loc, memrefType);

    // Populate the memref with key indices before the loop
    for (int64_t i = 0; i < numKeys; i++) {
      Value keyIndex = loadOps[i].getIndex();

      // Rematerialize the index if needed
      Value materializedIndex = rematerializeIndex(builder, loc, keyIndex);
      if (!materializedIndex) {
        LLVM_DEBUG(llvm::dbgs() << "  Failed to rematerialize index for key "
                                << i << ", skipping batch\n");
        return;
      }

      // Ensure it's index type
      if (!materializedIndex.getType().isIndex()) {
        if (materializedIndex.getType().isInteger(64)) {
          materializedIndex = builder.create<arith::IndexCastOp>(
              loc, indexType, materializedIndex);
        } else if (materializedIndex.getType().isInteger()) {
          materializedIndex = builder.create<arith::IndexCastOp>(
              loc, indexType, materializedIndex);
        }
      }

      Value memrefIdx = builder.create<arith::ConstantIndexOp>(loc, i);
      builder.create<memref::StoreOp>(loc, materializedIndex, keyIndicesMemref,
                                      ValueRange{memrefIdx});
    }

    // Insert prefetch inside the loop
    // We need to handle the case where numKeys > numIters by prefetching
    // multiple keys per iteration in sequential order
    builder.setInsertionPointToStart(forOp.getBody());
    Value loopIV = forOp.getInductionVar();

    // Get loop bounds to calculate how many keys per iteration
    int64_t lb = forOp.getConstantLowerBound();
    int64_t ub = forOp.getConstantUpperBound();
    int64_t step = forOp.getStep().getSExtValue();
    int64_t numIters = (ub - lb) / step;

    // Calculate how many keys to prefetch per iteration
    int64_t keysPerIter = (numKeys + numIters - 1) / numIters;  // ceil division

    LLVM_DEBUG(llvm::dbgs() << "  Prefetching " << keysPerIter
                            << " keys per iteration sequentially\n");

    // Prefetch keys in sequential order:
    // Iteration i prefetches keys[i*keysPerIter, i*keysPerIter+1, ...,
    // i*keysPerIter+keysPerIter-1]
    for (int64_t offset = 0; offset < keysPerIter; offset++) {
      // Calculate the key index: iv * keysPerIter + offset
      AffineExpr d0 = getAffineDimExpr(0, builder.getContext());

      // Create condition: (iv * keysPerIter + offset) < numKeys
      // In affine form: numKeys - 1 - offset - iv * keysPerIter >= 0
      // This directly expresses the bounds check without rounding issues

      // Create the affine constraint
      IntegerSet condSet = IntegerSet::get(
          1, 0, {numKeys - 1 - offset - d0 * keysPerIter}, {false});

      auto ifOp = builder.create<affine::AffineIfOp>(
          loc, condSet, ValueRange{loopIV}, /*withElseRegion=*/false);

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(ifOp.getThenBlock());

      // Calculate memref index = iv * keysPerIter + offset
      Value memrefIndex;
      if (keysPerIter == 1 && offset == 0) {
        // Simple case: just use iv
        memrefIndex = loopIV;
      } else {
        // Use affine.apply to compute iv * keysPerIter + offset
        AffineMap indexMap = AffineMap::get(1, 0, d0 * keysPerIter + offset,
                                            builder.getContext());
        memrefIndex = builder.create<affine::AffineApplyOp>(loc, indexMap,
                                                            ValueRange{loopIV});
      }

      // Load the key index from the memref
      Value keyIndex = builder.create<memref::LoadOp>(loc, keyIndicesMemref,
                                                      ValueRange{memrefIndex});

      // Cast to i64 for prefetch
      Value keyIndexI64 = builder.create<arith::IndexCastOp>(
          loc, builder.getI64Type(), keyIndex);

      builder.create<PrefetchKeyOp>(loc, keyIndexI64);

      // InsertionGuard automatically restores insertion point when it goes out
      // of scope
    }

    // Mark all loads as batched so we don't process them individually
    for (auto loadOp : loadOps) {
      loadOp->setAttr("kmrt.prefetch_batched", builder.getUnitAttr());
    }

    LLVM_DEBUG(llvm::dbgs() << "  Inserted batched prefetch loop\n");
  }

  // Check if a prefetch for this index already exists nearby
  bool hasPrefetchNearby(Operation* startOp, Value indexValue,
                         int lookback = 20) {
    Operation* current = startOp;
    for (int i = 0; i < lookback && current; i++) {
      current = current->getPrevNode();
      if (!current) break;

      if (auto prefetchOp = dyn_cast<PrefetchKeyOp>(current)) {
        if (prefetchOp.getIndex() == indexValue) {
          return true;
        }
      }
    }
    return false;
  }

  // Rematerialize an index value at the current builder insertion point
  // Returns nullptr if rematerialization is not possible
  Value rematerializeIndex(OpBuilder& builder, Location loc, Value indexValue) {
    // If it's a constant, create a new one
    if (auto constOp = indexValue.getDefiningOp<arith::ConstantOp>()) {
      return builder.create<arith::ConstantOp>(loc, constOp.getValue());
    }
    if (auto constIndexOp =
            indexValue.getDefiningOp<arith::ConstantIndexOp>()) {
      return builder.create<arith::ConstantIndexOp>(loc, constIndexOp.value());
    }

    // If it's an index_cast, rematerialize the input and cast it
    if (auto castOp = indexValue.getDefiningOp<arith::IndexCastOp>()) {
      Value inner = rematerializeIndex(builder, loc, castOp.getIn());
      if (!inner) return nullptr;
      return builder.create<arith::IndexCastOp>(loc, castOp.getType(), inner);
    }

    // If it's an affine.apply, rematerialize it if all operands are available
    if (auto applyOp = indexValue.getDefiningOp<affine::AffineApplyOp>()) {
      SmallVector<Value> newOperands;
      Block* insertBlock = builder.getInsertionBlock();

      for (Value operand : applyOp.getOperands()) {
        // Check if operand is available at insertion point
        bool operandAvailable = false;
        if (auto definingOp = operand.getDefiningOp()) {
          // Check if defining op is in a dominating block or earlier in the
          // same block
          Block* defBlock = definingOp->getBlock();
          if (defBlock == insertBlock) {
            // Same block - check if it's before the insertion point
            auto insertIt = builder.getInsertionPoint();
            if (insertIt != insertBlock->end()) {
              operandAvailable = definingOp->isBeforeInBlock(&*insertIt);
            } else {
              operandAvailable = true;  // Inserting at end, so it's available
            }
          } else {
            // Different blocks - check block dominance
            operandAvailable = domInfo->dominates(defBlock, insertBlock);
          }
        } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
          // Block argument - must dominate
          operandAvailable =
              domInfo->dominates(blockArg.getOwner(), insertBlock);
        }

        if (!operandAvailable) {
          // Try to rematerialize the operand
          Value rematerialized = rematerializeIndex(builder, loc, operand);
          if (!rematerialized) return nullptr;
          newOperands.push_back(rematerialized);
        } else {
          newOperands.push_back(operand);
        }
      }

      return builder.create<affine::AffineApplyOp>(loc, applyOp.getAffineMap(),
                                                   newOperands);
    }

    // If it's a value that dominates, use it directly
    if (auto definingOp = indexValue.getDefiningOp()) {
      Block* defBlock = definingOp->getBlock();
      Block* insertBlock = builder.getInsertionBlock();
      if (defBlock == insertBlock) {
        auto insertIt = builder.getInsertionPoint();
        if (insertIt != insertBlock->end() &&
            definingOp->isBeforeInBlock(&*insertIt)) {
          return indexValue;
        } else if (insertIt == insertBlock->end()) {
          return indexValue;
        }
      } else if (domInfo->dominates(defBlock, insertBlock)) {
        return indexValue;
      }
    } else if (auto blockArg = dyn_cast<BlockArgument>(indexValue)) {
      if (domInfo->dominates(blockArg.getOwner(),
                             builder.getInsertionBlock())) {
        return indexValue;
      }
    }

    // Cannot rematerialize
    return nullptr;
  }

  // Insert prefetch for a single load_key operation (non-loop case)
  void insertPrefetchForLoad(LoadKeyOp loadOp) {
    Value indexValue = loadOp.getIndex();

    // Check if there's already a prefetch for this key nearby
    if (hasPrefetchNearby(loadOp.getOperation(), indexValue)) {
      LLVM_DEBUG(llvm::dbgs() << "KMRTKeyPrefetching: Skipping - prefetch "
                                 "already exists nearby\n");
      return;
    }

    // Check if there's an affine.for loop immediately before this load_key
    // If so, insert prefetch in the last iteration of that loop instead
    Operation* prevOp = loadOp->getPrevNode();
    affine::AffineForOp precedingLoop = nullptr;

    // Look backwards for the most recent affine.for
    while (prevOp) {
      if (auto forOp = dyn_cast<affine::AffineForOp>(prevOp)) {
        precedingLoop = forOp;
        break;
      }
      // Skip operations that don't produce significant work
      if (isa<arith::ConstantOp, arith::ConstantIndexOp>(prevOp)) {
        prevOp = prevOp->getPrevNode();
        continue;
      }
      break;  // Stop at first non-trivial operation
    }

    if (precedingLoop) {
      // Check if index dominates the preceding loop before trying to insert
      // inside it
      Value indexValue = loadOp.getIndex();
      bool canUseIndex = true;

      // Helper to check if a value is defined before the loop (dominates it)
      auto checkValueDominates = [&](Value val) -> bool {
        if (auto defOp = val.getDefiningOp()) {
          Block* defBlock = defOp->getBlock();
          Block* loopParentBlock = precedingLoop.getOperation()->getBlock();

          // If defined in a different block, check block dominance
          if (defBlock != loopParentBlock) {
            return domInfo->dominates(defBlock, loopParentBlock);
          }

          // Same block - check execution order (must be before the loop)
          return defOp->isBeforeInBlock(precedingLoop.getOperation());
        } else if (auto blockArg = dyn_cast<BlockArgument>(val)) {
          return domInfo->dominates(blockArg.getOwner(),
                                    precedingLoop.getOperation()->getBlock());
        }
        return false;
      };

      // Recursively check if the index and its dependencies can be used
      SmallVector<Value> worklist = {indexValue};
      DenseSet<Value> visited;

      while (!worklist.empty() && canUseIndex) {
        Value current = worklist.pop_back_val();
        if (!visited.insert(current).second) continue;

        // Constants can always be rematerialized, skip dominance check
        if (auto defOp = current.getDefiningOp()) {
          if (isa<arith::ConstantOp, arith::ConstantIndexOp>(defOp)) {
            continue;  // No need to check dominance or operands for constants
          }
        }

        // Check if non-constant value dominates
        if (!checkValueDominates(current)) {
          canUseIndex = false;
          break;
        }

        // Add operands to worklist
        if (auto defOp = current.getDefiningOp()) {
          if (isa<affine::AffineApplyOp, arith::IndexCastOp>(defOp)) {
            for (Value operand : defOp->getOperands()) {
              worklist.push_back(operand);
            }
          }
        }
      }

      if (canUseIndex && precedingLoop.hasConstantBounds()) {
        // Index dominates, we can insert inside the loop
        insertPrefetchAtIterationBasedOnCost(precedingLoop, loadOp);
        return;
      }
      // Otherwise, fall through to the normal backward walk
    }

    // Original logic: Walk backwards accumulating costs
    // If we hit a loop, place prefetch inside that loop instead of jumping back
    // over it
    int64_t accumulatedCost = 0;
    Operation* current = loadOp.getOperation();
    Operation* placementPoint = current;

    while (current && accumulatedCost < prefetchThreshold.getValue()) {
      current = current->getPrevNode();
      if (!current) break;

      // If we encounter a loop, check if we can place the prefetch inside it
      if (auto forOp = dyn_cast<affine::AffineForOp>(current)) {
        // Only try to place prefetch inside if loop has constant bounds
        if (!forOp.hasConstantBounds()) {
          LLVM_DEBUG(llvm::dbgs()
                     << "KMRTKeyPrefetching: Loop has non-constant bounds, "
                        "skipping\n");
          accumulatedCost += 100;  // Conservative estimate
          continue;
        }

        // Check if the index value dominates the loop entry
        // This is a quick check to avoid dominance issues
        Value indexValue = loadOp.getIndex();
        bool canUseIndex = true;

        // Helper to check if a value is defined before the loop (dominates it)
        auto checkValueDominates = [&](Value val) -> bool {
          if (auto defOp = val.getDefiningOp()) {
            Block* defBlock = defOp->getBlock();
            Block* loopParentBlock = forOp.getOperation()->getBlock();

            // If defined in a different block, check block dominance
            if (defBlock != loopParentBlock) {
              return domInfo->dominates(defBlock, loopParentBlock);
            }

            // Same block - check execution order (must be before the loop)
            return defOp->isBeforeInBlock(forOp.getOperation());
          } else if (auto blockArg = dyn_cast<BlockArgument>(val)) {
            return domInfo->dominates(blockArg.getOwner(),
                                      forOp.getOperation()->getBlock());
          }
          return false;
        };

        // Recursively check if the index and its dependencies can be used
        SmallVector<Value> worklist = {indexValue};
        DenseSet<Value> visited;

        while (!worklist.empty() && canUseIndex) {
          Value current = worklist.pop_back_val();
          if (!visited.insert(current).second) continue;

          // Constants can always be rematerialized, skip dominance check
          if (auto defOp = current.getDefiningOp()) {
            if (isa<arith::ConstantOp, arith::ConstantIndexOp>(defOp)) {
              continue;  // No need to check dominance or operands for constants
            }
          }

          // Check if non-constant value dominates
          if (!checkValueDominates(current)) {
            canUseIndex = false;
            break;
          }

          // Add operands to worklist
          if (auto defOp = current.getDefiningOp()) {
            if (isa<affine::AffineApplyOp, arith::IndexCastOp>(defOp)) {
              for (Value operand : defOp->getOperands()) {
                worklist.push_back(operand);
              }
            }
          }
        }

        if (canUseIndex) {
          // Index dominates the loop, we can place prefetch inside
          LLVM_DEBUG(
              llvm::dbgs()
              << "KMRTKeyPrefetching: Found loop while walking backwards, "
                 "placing prefetch inside loop\n");
          insertPrefetchAtIterationBasedOnCost(forOp, loadOp);
          return;
        } else {
          // Index depends on values defined after the loop
          // We can't go back past this loop, so stop here and place prefetch
          // close to load
          LLVM_DEBUG(llvm::dbgs()
                     << "KMRTKeyPrefetching: Index doesn't dominate loop, "
                        "stopping backward walk\n");
          placementPoint = loadOp.getOperation();
          break;
        }
      }

      int64_t opCost = getOperationCost(current);
      if (opCost == 0) continue;  // Skip zero-cost operations

      accumulatedCost += opCost;

      if (accumulatedCost >= prefetchThreshold.getValue()) {
        placementPoint = current;
        break;
      }

      placementPoint = current;
    }

    // Insert prefetch at placement point
    OpBuilder builder(placementPoint);
    Location loc = loadOp.getLoc();

    // Rematerialize the index value at the prefetch location
    Value prefetchIndex = rematerializeIndex(builder, loc, indexValue);
    if (!prefetchIndex) {
      LLVM_DEBUG(llvm::dbgs()
                 << "KMRTKeyPrefetching: Skipping prefetch - "
                    "cannot rematerialize index at placement point\n");
      return;
    }

    // Ensure index is i64 for prefetch_key
    if (!prefetchIndex.getType().isInteger(64)) {
      if (prefetchIndex.getType().isIndex()) {
        prefetchIndex = builder.create<arith::IndexCastOp>(
            loc, builder.getI64Type(), prefetchIndex);
      } else if (prefetchIndex.getType().isInteger()) {
        // Handle other integer types (e.g., i32, i16)
        auto intType = cast<IntegerType>(prefetchIndex.getType());
        if (intType.getWidth() < 64) {
          prefetchIndex = builder.create<arith::ExtSIOp>(
              loc, builder.getI64Type(), prefetchIndex);
        } else if (intType.getWidth() > 64) {
          prefetchIndex = builder.create<arith::TruncIOp>(
              loc, builder.getI64Type(), prefetchIndex);
        }
      }
    }

    builder.create<PrefetchKeyOp>(loc, prefetchIndex);

    LLVM_DEBUG(llvm::dbgs() << "KMRTKeyPrefetching: Inserted prefetch with "
                               "accumulated cost="
                            << accumulatedCost << "\n");
  }

  // Calculate the cost of a single iteration of a loop
  // For nested loops, multiply the inner block cost by the loop bounds
  // Only consider outer-level affine.if, not deeply nested structures
  int64_t calculateLoopIterationCost(affine::AffineForOp forOp) {
    int64_t totalCost = 0;

    for (Operation& op : forOp.getBody()->without_terminator()) {
      // For nested loops, multiply inner block cost by iteration count
      if (auto nestedForOp = dyn_cast<affine::AffineForOp>(&op)) {
        if (nestedForOp.hasConstantBounds()) {
          int64_t lb = nestedForOp.getConstantLowerBound();
          int64_t ub = nestedForOp.getConstantUpperBound();
          int64_t step = nestedForOp.getStep().getSExtValue();
          int64_t iterCount = (ub - lb + step - 1) / step;

          // Get cost of one iteration of nested loop
          int64_t nestedIterCost = 0;
          for (Operation& nestedOp :
               nestedForOp.getBody()->without_terminator()) {
            nestedIterCost += getOperationCost(&nestedOp);
          }

          totalCost += nestedIterCost * iterCount;
          LLVM_DEBUG(llvm::dbgs() << "    Nested loop: " << iterCount
                                  << " iters * " << nestedIterCost << " cost = "
                                  << (nestedIterCost * iterCount) << "\n");
        } else {
          // For non-constant bounds, estimate conservatively
          int64_t nestedIterCost = 0;
          for (Operation& nestedOp :
               nestedForOp.getBody()->without_terminator()) {
            nestedIterCost += getOperationCost(&nestedOp);
          }
          // Assume 10 iterations as a conservative estimate
          totalCost += nestedIterCost * 10;
          LLVM_DEBUG(llvm::dbgs()
                     << "    Nested loop (symbolic): est 10 iters * "
                     << nestedIterCost << " = " << (nestedIterCost * 10)
                     << "\n");
        }
      } else if (auto ifOp = dyn_cast<affine::AffineIfOp>(&op)) {
        // For affine.if at outer level, take max of both branches
        int64_t thenCost = 0;
        for (Operation& thenOp : ifOp.getThenBlock()->without_terminator()) {
          thenCost += getOperationCost(&thenOp);
        }

        int64_t elseCost = 0;
        if (ifOp.hasElse()) {
          for (Operation& elseOp : ifOp.getElseBlock()->without_terminator()) {
            elseCost += getOperationCost(&elseOp);
          }
        }

        totalCost += std::max(thenCost, elseCost);
        LLVM_DEBUG(llvm::dbgs()
                   << "    Affine.if: max(" << thenCost << ", " << elseCost
                   << ") = " << std::max(thenCost, elseCost) << "\n");
      } else {
        // Regular operation
        int64_t opCost = getOperationCost(&op);
        totalCost += opCost;
      }
    }

    return totalCost;
  }

  // Insert prefetch at a specific iteration of a loop based on cost analysis
  void insertPrefetchAtIterationBasedOnCost(affine::AffineForOp forOp,
                                            LoadKeyOp loadOp) {
    if (!forOp.hasConstantBounds()) {
      LLVM_DEBUG(llvm::dbgs() << "KMRTKeyPrefetching: Cannot prefetch based on "
                                 "cost - non-constant bounds\n");
      return;
    }

    int64_t lb = forOp.getConstantLowerBound();
    int64_t ub = forOp.getConstantUpperBound();
    int64_t step = forOp.getStep().getSExtValue();
    int64_t iterCount = (ub - lb) / step;

    // Calculate cost per iteration
    int64_t iterCost = calculateLoopIterationCost(forOp);

    LLVM_DEBUG(llvm::dbgs() << "KMRTKeyPrefetching: Loop cost analysis:\n");
    LLVM_DEBUG(llvm::dbgs() << "  Iteration cost=" << iterCost << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  Iterations=" << iterCount << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "  Total loop cost=" << (iterCost * iterCount) << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  Prefetch threshold="
                            << prefetchThreshold.getValue() << "\n");

    // Determine which iteration to place the prefetch in
    // We want: (remaining iterations * iterCost) >= prefetchThreshold
    // So: remaining iters >= prefetchThreshold / iterCost
    // Which means: targetIter = totalIters - (threshold / iterCost)

    int64_t targetRemainingCost = prefetchThreshold.getValue();
    int64_t remainingIters =
        (targetRemainingCost + iterCost - 1) / iterCost;  // Round up
    int64_t targetIterOffset = iterCount - remainingIters;

    // Clamp to valid range [0, iterCount-1]
    if (targetIterOffset < 0) {
      targetIterOffset = 0;
    } else if (targetIterOffset >= iterCount) {
      targetIterOffset = iterCount - 1;
    }

    int64_t targetIteration = lb + targetIterOffset * step;

    LLVM_DEBUG(llvm::dbgs() << "  Target iteration: " << targetIteration
                            << " (offset=" << targetIterOffset << ")\n");
    LLVM_DEBUG(llvm::dbgs()
               << "  Remaining cost from prefetch: "
               << ((iterCount - targetIterOffset) * iterCost) << "\n");

    OpBuilder builder(&forOp.getBody()->front());
    Location loc = forOp.getLoc();
    Value loopIV = forOp.getInductionVar();
    Value loadIndex = loadOp.getIndex();

    // Create condition: iv == targetIteration
    AffineExpr d0 = getAffineDimExpr(0, builder.getContext());
    IntegerSet condSet = IntegerSet::get(
        1, 0, {d0 - targetIteration, targetIteration - d0}, {false, false});

    auto ifOp = builder.create<affine::AffineIfOp>(
        loc, condSet, ValueRange{loopIV}, /*withElseRegion=*/false);

    // Inside the if, rematerialize and prefetch the key
    builder.setInsertionPointToStart(ifOp.getThenBlock());

    Value prefetchIndex = rematerializeIndex(builder, loc, loadIndex);
    if (!prefetchIndex) {
      LLVM_DEBUG(llvm::dbgs()
                 << "KMRTKeyPrefetching: Cannot rematerialize key index\n");
      return;
    }

    // Ensure it's i64
    if (!prefetchIndex.getType().isInteger(64)) {
      if (prefetchIndex.getType().isIndex()) {
        prefetchIndex = builder.create<arith::IndexCastOp>(
            loc, builder.getI64Type(), prefetchIndex);
      }
    }

    builder.create<PrefetchKeyOp>(loc, prefetchIndex);
    LLVM_DEBUG(llvm::dbgs()
               << "KMRTKeyPrefetching: Inserted prefetch at iteration "
               << targetIteration << "\n");
  }

  // Process an affine loop containing load_key operations
  void processAffineLoop(affine::AffineForOp forOp) {
    // Only process each loop once
    if (forOp->hasAttr("kmrt.prefetch_processed")) {
      return;
    }
    forOp->setAttr("kmrt.prefetch_processed",
                   UnitAttr::get(forOp.getContext()));

    // Check if this loop has any load_key operations (including nested)
    bool hasLoads = false;
    forOp.walk([&](LoadKeyOp loadOp) {
      hasLoads = true;
      return WalkResult::interrupt();
    });

    if (!hasLoads) {
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "KMRTKeyPrefetching: Processing affine loop "
                               "with load_key operations\n");

    // Create a prefetch loop structure before this loop
    createPrefetchLoopStructure(forOp);
  }

  // Recursively clone a block's operations, replacing load_key with
  // prefetch_key and preserving affine.for and affine.if control flow
  // structures
  void cloneBlockForPrefetch(OpBuilder& builder, Block* sourceBlock,
                             IRMapping& mapping,
                             DenseSet<Value>& prefetchedKeys) {
    for (Operation& op : *sourceBlock) {
      if (auto loadOp = dyn_cast<LoadKeyOp>(&op)) {
        // Replace load_key with prefetch_key
        Value indexValue = loadOp.getIndex();
        Value mappedIndex = mapping.lookupOrDefault(indexValue);

        // Check if we've already prefetched this key
        if (prefetchedKeys.contains(mappedIndex)) {
          LLVM_DEBUG(llvm::dbgs() << "  Skipping duplicate prefetch\n");
          continue;
        }

        // Ensure index is i64 for prefetch_key
        if (!mappedIndex.getType().isInteger(64)) {
          if (mappedIndex.getType().isIndex()) {
            mappedIndex = builder.create<arith::IndexCastOp>(
                loadOp.getLoc(), builder.getI64Type(), mappedIndex);
          } else if (mappedIndex.getType().isInteger()) {
            auto intType = cast<IntegerType>(mappedIndex.getType());
            if (intType.getWidth() < 64) {
              mappedIndex = builder.create<arith::ExtSIOp>(
                  loadOp.getLoc(), builder.getI64Type(), mappedIndex);
            } else if (intType.getWidth() > 64) {
              mappedIndex = builder.create<arith::TruncIOp>(
                  loadOp.getLoc(), builder.getI64Type(), mappedIndex);
            }
          }
        }

        builder.create<PrefetchKeyOp>(loadOp.getLoc(), mappedIndex);
        prefetchedKeys.insert(mappedIndex);

        LLVM_DEBUG(llvm::dbgs() << "  Created prefetch_key\n");

      } else if (auto nestedForOp = dyn_cast<affine::AffineForOp>(&op)) {
        // Clone the nested loop structure
        LLVM_DEBUG(llvm::dbgs() << "  Cloning nested affine.for\n");

        // Map operands for bounds
        SmallVector<Value> lowerBoundOperands;
        for (Value operand : nestedForOp.getLowerBoundOperands()) {
          lowerBoundOperands.push_back(mapping.lookupOrDefault(operand));
        }
        SmallVector<Value> upperBoundOperands;
        for (Value operand : nestedForOp.getUpperBoundOperands()) {
          upperBoundOperands.push_back(mapping.lookupOrDefault(operand));
        }

        // Create the prefetch loop
        int64_t step = nestedForOp.getStep().getSExtValue();
        auto prefetchLoop = builder.create<affine::AffineForOp>(
            nestedForOp.getLoc(), lowerBoundOperands,
            nestedForOp.getLowerBoundMap(), upperBoundOperands,
            nestedForOp.getUpperBoundMap(), step);

        // Map the induction variable
        mapping.map(nestedForOp.getInductionVar(),
                    prefetchLoop.getInductionVar());

        // Recursively clone the loop body
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(prefetchLoop.getBody());
        cloneBlockForPrefetch(builder, nestedForOp.getBody(), mapping,
                              prefetchedKeys);

      } else if (auto ifOp = dyn_cast<affine::AffineIfOp>(&op)) {
        // Clone the affine.if structure
        LLVM_DEBUG(llvm::dbgs() << "  Cloning affine.if\n");

        // Map operands for the condition
        SmallVector<Value> operands;
        for (Value operand : ifOp.getOperands()) {
          operands.push_back(mapping.lookupOrDefault(operand));
        }

        // Create the prefetch if
        auto prefetchIf = builder.create<affine::AffineIfOp>(
            ifOp.getLoc(), ifOp.getIntegerSet(), operands, ifOp.hasElse());

        // Clone the then block
        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(prefetchIf.getThenBlock());
          cloneBlockForPrefetch(builder, ifOp.getThenBlock(), mapping,
                                prefetchedKeys);
        }

        // Clone the else block if present
        if (ifOp.hasElse()) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(prefetchIf.getElseBlock());
          cloneBlockForPrefetch(builder, ifOp.getElseBlock(), mapping,
                                prefetchedKeys);
        }

      } else if (isa<affine::AffineApplyOp, arith::ConstantOp,
                     arith::ConstantIndexOp, arith::IndexCastOp, arith::ExtSIOp,
                     arith::TruncIOp>(&op)) {
        // Clone index computation operations that might be needed for computing
        // key indices
        LLVM_DEBUG(llvm::dbgs() << "  Cloning index computation op: "
                                << op.getName() << "\n");

        // Clone the operation with mapped operands
        SmallVector<Value> mappedOperands;
        for (Value operand : op.getOperands()) {
          mappedOperands.push_back(mapping.lookupOrDefault(operand));
        }

        Operation* clonedOp = builder.clone(op, mapping);

        // Update mapping for results
        for (auto [origResult, clonedResult] :
             llvm::zip(op.getResults(), clonedOp->getResults())) {
          mapping.map(origResult, clonedResult);
        }
      }
      // Skip all other operations (computational ops, clear_key, etc.)
      // We only care about control flow, index computation, and load_key for
      // prefetching
    }
  }

  // Structure to track conditional context of a load_key
  struct ConditionalInfo {
    affine::AffineIfOp ifOp;
    bool inThenBranch;  // true if in then branch, false if in else branch
  };

  struct LoadKeyContext {
    LoadKeyOp loadOp;
    // Chain of enclosing affine.if operations (from outermost to innermost)
    SmallVector<ConditionalInfo> conditionalChain;

    // Legacy accessors for backward compatibility
    affine::AffineIfOp enclosingIf() const {
      return conditionalChain.empty() ? nullptr : conditionalChain.back().ifOp;
    }
    bool inThenBranch() const {
      return conditionalChain.empty() ? false
                                      : conditionalChain.back().inThenBranch;
    }
  };

  // Create integrated prefetching for a loop (prefetch next iteration inside
  // loop)
  void createPrefetchLoopStructure(affine::AffineForOp forOp) {
    OpBuilder builder(forOp);

    LLVM_DEBUG(llvm::dbgs() << "Creating integrated prefetch structure\n");

    // Collect all load_key operations in this loop (not in nested loops)
    // This includes load_key inside affine.if blocks
    SmallVector<LoadKeyContext> directLoadOps;

    // Helper to recursively collect load_key from a block, stopping at nested
    // loops conditionalChain: chain of enclosing affine.if operations (from
    // outermost to innermost)
    std::function<void(Block*, SmallVector<ConditionalInfo>&)>
        collectFromBlock =
            [&](Block* block, SmallVector<ConditionalInfo>& conditionalChain) {
              for (Operation& op : block->without_terminator()) {
                if (auto loadOp = dyn_cast<LoadKeyOp>(&op)) {
                  // Add this load_key operation with its full conditional
                  // context llvm::errs() << "[KeyPrefetch] Found load_key with
                  // " << conditionalChain.size() << " conditionals in chain\n";
                  LoadKeyContext ctx;
                  ctx.loadOp = loadOp;
                  ctx.conditionalChain = conditionalChain;
                  directLoadOps.push_back(ctx);
                } else if (auto ifOp = dyn_cast<affine::AffineIfOp>(&op)) {
                  // Recursively search inside affine.if blocks, extending the
                  // chain
                  SmallVector<ConditionalInfo> thenChain = conditionalChain;
                  thenChain.push_back({ifOp, true});
                  collectFromBlock(ifOp.getThenBlock(), thenChain);

                  if (ifOp.hasElse()) {
                    SmallVector<ConditionalInfo> elseChain = conditionalChain;
                    elseChain.push_back({ifOp, false});
                    collectFromBlock(ifOp.getElseBlock(), elseChain);
                  }
                } else if (isa<affine::AffineForOp>(&op)) {
                  // Don't recurse into nested affine.for loops - they'll be
                  // processed separately Just skip them
                }
              }
            };

    SmallVector<ConditionalInfo> emptyChain;
    collectFromBlock(forOp.getBody(), emptyChain);

    // IMPORTANT: Collect nested loops BEFORE modifying the IR
    // This prevents infinite recursion when we insert new operations
    SmallVector<affine::AffineForOp> nestedLoops;
    for (Operation& op : forOp.getBody()->without_terminator()) {
      if (auto nestedForOp = dyn_cast<affine::AffineForOp>(&op)) {
        nestedLoops.push_back(nestedForOp);
      }
    }

    // For each load_key that depends on the loop IV, insert prefetches
    if (!directLoadOps.empty()) {
      for (auto& ctx : directLoadOps) {
        insertIntegratedPrefetch(forOp, ctx);
      }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "No direct load_key ops in loop body\n");
    }

    // Process nested loops (collected before we modified anything)
    for (auto nestedForOp : nestedLoops) {
      createPrefetchLoopStructure(nestedForOp);
    }

    LLVM_DEBUG(llvm::dbgs() << "Integrated prefetch structure created\n");
  }

  // Helper to insert a prefetch, wrapping it in the load's conditional if
  // needed conditionIV: the value to use in the condition (could be current IV
  // or next IV)
  void insertConditionalPrefetch(OpBuilder& builder, Location loc,
                                 Value prefetchIndex, LoadKeyContext& ctx,
                                 Value conditionIV) {
    // Ensure prefetchIndex is i64
    if (!prefetchIndex.getType().isInteger(64)) {
      if (prefetchIndex.getType().isIndex()) {
        prefetchIndex = builder.create<arith::IndexCastOp>(
            loc, builder.getI64Type(), prefetchIndex);
      }
    }

    // If the load is conditional, wrap the prefetch in the full chain of
    // conditions
    if (!ctx.conditionalChain.empty()) {
      llvm::errs() << "[KeyPrefetch] Building nested conditional prefetch with "
                   << ctx.conditionalChain.size() << " levels\n";
      LLVM_DEBUG(llvm::dbgs() << "  Building nested conditional prefetch with "
                              << ctx.conditionalChain.size() << " levels\n");
      // Build nested affine.if structure from outermost to innermost
      for (size_t level = 0; level < ctx.conditionalChain.size(); ++level) {
        auto& condInfo = ctx.conditionalChain[level];
        llvm::errs() << "[KeyPrefetch]   Creating level " << level
                     << " (inThenBranch=" << condInfo.inThenBranch << ")\n";
        LLVM_DEBUG(llvm::dbgs()
                   << "    Creating level " << level
                   << " (inThenBranch=" << condInfo.inThenBranch << ")\n");
        IntegerSet condSet = condInfo.ifOp.getIntegerSet();

        // Build operands for the condition, replacing the loop IV with
        // conditionIV
        SmallVector<Value> condOperands;
        for (Value operand : condInfo.ifOp.getOperands()) {
          // The first operand in affine.if is typically the loop IV we want to
          // replace
          if (condOperands.empty() && isa<BlockArgument>(operand)) {
            condOperands.push_back(conditionIV);
          } else {
            condOperands.push_back(operand);
          }
        }

        // Create affine.if with then/else based on which branch the load is in
        auto prefetchIf = builder.create<affine::AffineIfOp>(
            loc, condSet, condOperands,
            /*withElseRegion=*/!condInfo.inThenBranch);

        // Move insertion point to the appropriate branch for the next level
        if (condInfo.inThenBranch) {
          builder.setInsertionPointToStart(prefetchIf.getThenBlock());
        } else {
          builder.setInsertionPointToStart(prefetchIf.getElseBlock());
        }
      }

      // Now insert the prefetch at the innermost level
      llvm::errs()
          << "[KeyPrefetch]   Inserting prefetch_key at innermost level\n";
      LLVM_DEBUG(llvm::dbgs()
                 << "    Inserting prefetch_key at innermost level\n");
      builder.create<PrefetchKeyOp>(loc, prefetchIndex);
    } else {
      llvm::errs() << "[KeyPrefetch] Inserting unconditional prefetch_key\n";
      LLVM_DEBUG(llvm::dbgs() << "  Inserting unconditional prefetch_key\n");
      builder.create<PrefetchKeyOp>(loc, prefetchIndex);
    }
  }

  // Insert prefetch for first iteration before loop, and prefetch for next
  // iteration inside loop
  void insertIntegratedPrefetch(affine::AffineForOp forOp,
                                LoadKeyContext& ctx) {
    LoadKeyOp loadOp = ctx.loadOp;
    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();
    Value loopIV = forOp.getInductionVar();
    Value loadIndex = loadOp.getIndex();

    LLVM_DEBUG(llvm::dbgs() << "Inserting integrated prefetch for load_key\n");
    if (!ctx.conditionalChain.empty()) {
      llvm::errs() << "[KeyPrefetch] Load is conditional with "
                   << ctx.conditionalChain.size() << " nested levels\n";
      for (size_t i = 0; i < ctx.conditionalChain.size(); ++i) {
        llvm::errs() << "[KeyPrefetch]   Level " << i << ": inThenBranch="
                     << ctx.conditionalChain[i].inThenBranch << "\n";
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "  Load is conditional with " << ctx.conditionalChain.size()
                 << " nested levels\n");
      for (size_t i = 0; i < ctx.conditionalChain.size(); ++i) {
        LLVM_DEBUG(llvm::dbgs()
                   << "    Level " << i << ": inThenBranch="
                   << ctx.conditionalChain[i].inThenBranch << "\n");
      }
    }

    // Check if the load index depends on the loop IV
    if (!indexDependsOnIV(loadIndex, loopIV)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Index doesn't depend on loop IV, skipping\n");
      return;
    }

    // 1. Prefetch the first iteration's key before the loop
    // NOTE: For conditional loads, we skip the before-loop prefetch for now
    // because the loop IV isn't available yet (dominance issue)
    // The first iteration's key will be prefetched during the loop's first
    // iteration
    if (ctx.conditionalChain.empty()) {
      // For constant lower bound, we can evaluate it directly
      if (forOp.hasConstantLowerBound()) {
        int64_t lb = forOp.getConstantLowerBound();
        Value firstIterIndex =
            computeIndexForConstantIV(builder, loc, loopIV, lb, loadIndex);

        if (firstIterIndex) {
          insertConditionalPrefetch(builder, loc, firstIterIndex, ctx, loopIV);
          LLVM_DEBUG(
              llvm::dbgs()
              << "  Inserted prefetch before loop for first iteration\n");
        }
      } else {
        // Symbolic lower bound - compute the first iteration value
        Value lowerBound = builder.create<affine::AffineApplyOp>(
            loc, forOp.getLowerBoundMap(), forOp.getLowerBoundOperands());

        Value firstIterIndex =
            computeIndexForLoopIV(builder, loc, loopIV, lowerBound, loadIndex);

        if (firstIterIndex) {
          insertConditionalPrefetch(builder, loc, firstIterIndex, ctx, loopIV);
          LLVM_DEBUG(llvm::dbgs() << "  Inserted prefetch before loop for "
                                     "first iteration (symbolic)\n");
        }
      }
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Skipping before-loop prefetch for conditional load\n");
    }

    // 2. Insert prefetch for next iteration at the beginning of the loop body
    builder.setInsertionPoint(&forOp.getBody()->front());

    // Create affine map for next iteration: iv + step
    int64_t step = forOp.getStep().getSExtValue();
    AffineExpr d0 = getAffineDimExpr(0, builder.getContext());
    AffineMap nextIVMap = AffineMap::get(1, 0, d0 + step, builder.getContext());

    // Create condition: iv + step < upper_bound
    if (forOp.hasConstantUpperBound()) {
      int64_t ub = forOp.getConstantUpperBound();

      // Create affine.if condition: iv < ub - step (or iv + step < ub)
      AffineExpr condition = d0.ceilDiv(1);  // Just d0
      // We want: d0 + step < ub, which is d0 < ub - step
      int64_t checkBound = ub - step;

      if (checkBound >= forOp.getConstantLowerBound()) {
        // Create set: d0 < checkBound
        // IntegerSet uses constraints of the form: expr >= 0
        // So d0 < checkBound becomes checkBound - d0 - 1 >= 0
        IntegerSet condSet =
            IntegerSet::get(1, 0, {checkBound - condition - 1}, {false});

        auto ifOp = builder.create<affine::AffineIfOp>(
            loc, condSet, ValueRange{loopIV}, /*withElseRegion=*/false);

        // Inside the if, compute and prefetch next iteration's key
        builder.setInsertionPointToStart(ifOp.getThenBlock());

        // Compute iv + step
        Value nextIV = builder.create<affine::AffineApplyOp>(
            loc, nextIVMap, ValueRange{loopIV});

        // Compute the key index for next iteration
        Value nextIterIndex =
            computeIndexForLoopIV(builder, loc, loopIV, nextIV, loadIndex);

        if (nextIterIndex) {
          insertConditionalPrefetch(builder, loc, nextIterIndex, ctx, nextIV);
          LLVM_DEBUG(llvm::dbgs()
                     << "  Inserted prefetch inside loop for next iteration\n");
        }
      }
    } else {
      // Symbolic upper bound
      // Create condition: iv + step < upper_bound
      // We need to build an affine.if with symbolic operands

      // Get upper bound operands
      SmallVector<Value> condOperands;
      condOperands.push_back(loopIV);
      for (Value operand : forOp.getUpperBoundOperands()) {
        condOperands.push_back(operand);
      }

      // Build constraint: iv + step < ub
      // This becomes: ub - iv - step - 1 >= 0
      // With d0 = iv, s0, s1, ... = upper bound operands
      AffineMap ubMap = forOp.getUpperBoundMap();
      // The upper bound expression
      AffineExpr ubExpr = ubMap.getResult(0);

      // Shift dimension and symbol indices
      // Original upper bound uses dimensions/symbols at certain positions
      // Now we have: d0 (iv), s0, s1, ... (original upper bound operands)
      unsigned numSymbols = forOp.getUpperBoundOperands().size();

      // Remap upper bound expression dimensions/symbols to symbols
      // The upper bound map might have dims and symbols
      SmallVector<AffineExpr> dimReplacements;
      for (unsigned i = 0; i < ubMap.getNumDims(); i++) {
        dimReplacements.push_back(getAffineSymbolExpr(i, builder.getContext()));
      }
      SmallVector<AffineExpr> symReplacements;
      for (unsigned i = 0; i < ubMap.getNumSymbols(); i++) {
        symReplacements.push_back(
            getAffineSymbolExpr(ubMap.getNumDims() + i, builder.getContext()));
      }
      AffineExpr remappedUB =
          ubExpr.replaceDimsAndSymbols(dimReplacements, symReplacements);

      // Now create: ub - d0 - step - 1 >= 0
      AffineExpr constraint = remappedUB - d0 - step - 1;

      IntegerSet condSet =
          IntegerSet::get(1, numSymbols, {constraint}, {false});

      auto ifOp = builder.create<affine::AffineIfOp>(loc, condSet, condOperands,
                                                     /*withElseRegion=*/false);

      // Inside the if, compute and prefetch next iteration's key
      builder.setInsertionPointToStart(ifOp.getThenBlock());

      // Compute iv + step
      Value nextIV = builder.create<affine::AffineApplyOp>(loc, nextIVMap,
                                                           ValueRange{loopIV});

      // Compute the key index for next iteration
      Value nextIterIndex =
          computeIndexForLoopIV(builder, loc, loopIV, nextIV, loadIndex);

      if (nextIterIndex) {
        insertConditionalPrefetch(builder, loc, nextIterIndex, ctx, nextIV);
        LLVM_DEBUG(llvm::dbgs() << "  Inserted prefetch inside loop for next "
                                   "iteration (symbolic)\n");
      }
    }
  }

  // Check if an index value depends on a loop induction variable
  bool indexDependsOnIV(Value index, Value iv) {
    // Direct use of IV
    if (index == iv) return true;

    // Check through index_cast
    if (auto castOp = index.getDefiningOp<arith::IndexCastOp>()) {
      return indexDependsOnIV(castOp.getIn(), iv);
    }

    // Check through affine.apply
    if (auto applyOp = index.getDefiningOp<affine::AffineApplyOp>()) {
      for (Value operand : applyOp.getOperands()) {
        if (indexDependsOnIV(operand, iv)) return true;
      }
    }

    return false;
  }

  // Compute index value for a specific constant IV value
  Value computeIndexForConstantIV(OpBuilder& builder, Location loc,
                                  Value originalIV, int64_t ivValue,
                                  Value loadIndex) {
    // If it's the IV directly, return the constant
    if (loadIndex == originalIV) {
      return builder.create<arith::ConstantIndexOp>(loc, ivValue);
    }

    // Check through index_cast
    if (auto castOp = loadIndex.getDefiningOp<arith::IndexCastOp>()) {
      Value innerIndex = computeIndexForConstantIV(builder, loc, originalIV,
                                                   ivValue, castOp.getIn());
      if (!innerIndex) return nullptr;

      if (innerIndex.getType() != castOp.getType()) {
        return builder.create<arith::IndexCastOp>(loc, castOp.getType(),
                                                  innerIndex);
      }
      return innerIndex;
    }

    // Check through affine.apply
    if (auto applyOp = loadIndex.getDefiningOp<affine::AffineApplyOp>()) {
      AffineMap map = applyOp.getAffineMap();
      SmallVector<Value> operands;

      for (Value operand : applyOp.getOperands()) {
        if (operand == originalIV) {
          // Replace IV with constant value
          operands.push_back(
              builder.create<arith::ConstantIndexOp>(loc, ivValue));
        } else {
          // Try to use the operand directly if it dominates
          Block* insertBlock = builder.getInsertionBlock();
          bool operandAvailable = false;

          if (auto definingOp = operand.getDefiningOp()) {
            Block* defBlock = definingOp->getBlock();
            operandAvailable = domInfo->dominates(defBlock, insertBlock);
          } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
            operandAvailable =
                domInfo->dominates(blockArg.getOwner(), insertBlock);
          }

          if (!operandAvailable) {
            // Try to rematerialize constant
            if (auto constOp = operand.getDefiningOp<arith::ConstantOp>()) {
              operands.push_back(
                  builder.create<arith::ConstantOp>(loc, constOp.getValue()));
            } else if (auto constIndexOp =
                           operand.getDefiningOp<arith::ConstantIndexOp>()) {
              operands.push_back(builder.create<arith::ConstantIndexOp>(
                  loc, constIndexOp.value()));
            } else {
              return nullptr;
            }
          } else {
            operands.push_back(operand);
          }
        }
      }

      return builder.create<affine::AffineApplyOp>(loc, map, operands);
    }

    // If it's a constant, just use it
    if (loadIndex.getDefiningOp<arith::ConstantOp>() ||
        loadIndex.getDefiningOp<arith::ConstantIndexOp>()) {
      return loadIndex;
    }

    return nullptr;
  }

  // Create a prefetch loop before the main loop
  void createPrefetchLoopBefore(affine::AffineForOp forOp,
                                ArrayRef<LoadKeyOp> loadOps) {
    if (loadOps.empty()) {
      return;
    }

    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();
    Value originalIV = forOp.getInductionVar();

    // Create a prefetch loop with the same bounds and step as the original loop
    // Convert step from APInt to int64_t
    int64_t step = forOp.getStep().getSExtValue();
    auto prefetchLoop = builder.create<affine::AffineForOp>(
        loc, forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
        forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(), step);

    // Set insertion point inside the prefetch loop
    builder.setInsertionPointToStart(prefetchLoop.getBody());
    Value prefetchIV = prefetchLoop.getInductionVar();

    // For each load_key in the original loop, create a prefetch in the prefetch
    // loop
    for (auto loadOp : loadOps) {
      Value loadIndex = loadOp.getIndex();

      // Compute the prefetch index by substituting the loop IV
      Value prefetchIndex = computeIndexForLoopIV(builder, loc, originalIV,
                                                  prefetchIV, loadIndex);

      // Ensure it's i64 for prefetch_key
      if (prefetchIndex) {
        // Cast to i64 if it's not already i64
        if (!prefetchIndex.getType().isInteger(64)) {
          if (prefetchIndex.getType().isIndex()) {
            prefetchIndex = builder.create<arith::IndexCastOp>(
                loc, builder.getI64Type(), prefetchIndex);
          } else if (prefetchIndex.getType().isInteger()) {
            auto intType = cast<IntegerType>(prefetchIndex.getType());
            if (intType.getWidth() < 64) {
              prefetchIndex = builder.create<arith::ExtSIOp>(
                  loc, builder.getI64Type(), prefetchIndex);
            } else if (intType.getWidth() > 64) {
              prefetchIndex = builder.create<arith::TruncIOp>(
                  loc, builder.getI64Type(), prefetchIndex);
            }
          }
        }

        builder.create<PrefetchKeyOp>(loc, prefetchIndex);

        LLVM_DEBUG(
            llvm::dbgs()
            << "KMRTKeyPrefetching: Created prefetch in prefetch loop\n");
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "KMRTKeyPrefetching: Created prefetch loop with "
                            << loadOps.size() << " prefetches\n");
  }

  // Compute the key index for the prefetch loop by substituting the original
  // loop IV with the prefetch loop IV
  Value computeIndexForLoopIV(OpBuilder& builder, Location loc,
                              Value originalIV, Value prefetchIV,
                              Value loadIndex) {
    // Try to trace back through index_cast
    if (auto castOp = loadIndex.getDefiningOp<arith::IndexCastOp>()) {
      Value innerIndex = computeIndexForLoopIV(builder, loc, originalIV,
                                               prefetchIV, castOp.getIn());
      if (!innerIndex) return nullptr;

      // Apply the same cast only if needed
      Type targetType = castOp.getType();
      if (innerIndex.getType() != targetType) {
        return builder.create<arith::IndexCastOp>(loc, targetType, innerIndex);
      }
      return innerIndex;
    }

    // If it's an affine.apply, we need to apply the same map with the prefetch
    // IV
    if (auto applyOp = loadIndex.getDefiningOp<affine::AffineApplyOp>()) {
      AffineMap map = applyOp.getAffineMap();
      SmallVector<Value> operands;

      // Get the insertion block for dominance checking
      Block* insertBlock = builder.getInsertionBlock();

      // Replace original loop IV with prefetch loop IV, keep other operands
      for (Value operand : applyOp.getOperands()) {
        if (operand == originalIV) {
          operands.push_back(prefetchIV);
        } else {
          // Check if this operand is available at the insertion point
          bool operandAvailable = false;
          if (auto definingOp = operand.getDefiningOp()) {
            Block* defBlock = definingOp->getBlock();
            if (defBlock == insertBlock) {
              auto insertIt = builder.getInsertionPoint();
              if (insertIt != insertBlock->end()) {
                operandAvailable = definingOp->isBeforeInBlock(&*insertIt);
              } else {
                operandAvailable = true;
              }
            } else {
              operandAvailable = domInfo->dominates(defBlock, insertBlock);
            }
          } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
            operandAvailable =
                domInfo->dominates(blockArg.getOwner(), insertBlock);
          }

          if (!operandAvailable) {
            // Operand doesn't dominate - try to rematerialize if it's a
            // constant
            if (auto definingOp = operand.getDefiningOp()) {
              if (auto constOp = dyn_cast<arith::ConstantOp>(definingOp)) {
                operands.push_back(
                    builder.create<arith::ConstantOp>(loc, constOp.getValue()));
              } else if (auto constIndexOp =
                             dyn_cast<arith::ConstantIndexOp>(definingOp)) {
                operands.push_back(builder.create<arith::ConstantIndexOp>(
                    loc, constIndexOp.value()));
              } else {
                LLVM_DEBUG(llvm::dbgs()
                           << "KMRTKeyPrefetching: Cannot prefetch - "
                              "affine.apply operand doesn't dominate\n");
                return nullptr;
              }
            } else {
              LLVM_DEBUG(llvm::dbgs()
                         << "KMRTKeyPrefetching: Cannot prefetch - "
                            "affine.apply block arg doesn't dominate\n");
              return nullptr;
            }
          } else {
            operands.push_back(operand);
          }
        }
      }

      // Apply the affine map with substituted operands
      return builder.create<affine::AffineApplyOp>(loc, map, operands);
    }

    // If it's the original loop IV directly, return the prefetch loop IV
    if (loadIndex == originalIV) {
      return prefetchIV;
    }

    // If it's a constant, we need to materialize a new constant to avoid
    // dominance violations
    if (auto constOp = loadIndex.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        int64_t constValue = intAttr.getInt();
        return builder.create<arith::ConstantOp>(
            loc, builder.getI64IntegerAttr(constValue));
      } else {
        // Fallback: clone the constant
        return builder.create<arith::ConstantOp>(loc, constOp.getValue());
      }
    } else if (auto constIndexOp =
                   loadIndex.getDefiningOp<arith::ConstantIndexOp>()) {
      int64_t constValue = constIndexOp.value();
      return builder.create<arith::ConstantOp>(
          loc, builder.getI64IntegerAttr(constValue));
    }

    // If it's an SSA value not derived from IV and not a constant, use it
    // directly (This value should dominate both loops, so it's safe to use)
    return loadIndex;
  }

  // ===================================================================
  // Verification Methods
  // ===================================================================

  // Extract abstract key index from a Value
  std::optional<AbstractKeyIndex> extractKeyIndex(Value indexValue) {
    AbstractKeyIndex result;

    // Try to extract from constant
    if (auto constOp = indexValue.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        result.constantValue = intAttr.getInt();
        return result;
      }
    }

    // Try to extract from index_cast
    if (auto castOp = indexValue.getDefiningOp<arith::IndexCastOp>()) {
      return extractKeyIndex(castOp.getIn());
    }

    // Try to extract from affine.apply
    if (auto applyOp = indexValue.getDefiningOp<affine::AffineApplyOp>()) {
      result.affineExpr = applyOp.getAffineMap().getResult(0);
      result.affineOperands.assign(applyOp.getOperands().begin(),
                                   applyOp.getOperands().end());
      return result;
    }

    // Direct use of induction variable
    if (auto blockArg = dyn_cast<BlockArgument>(indexValue)) {
      // Create identity affine expression: d0 -> d0
      result.affineExpr = getAffineDimExpr(0, blockArg.getContext());
      result.affineOperands.push_back(indexValue);
      return result;
    }

    return std::nullopt;
  }

  // Verify a single block with given initial state
  LogicalResult verifyBlock(Block* block, VerificationState& state,
                            DenseMap<Value, int64_t>& ivValues) {
    for (Operation& op : *block) {
      // Handle prefetch_key operations
      if (auto prefetchOp = dyn_cast<PrefetchKeyOp>(&op)) {
        auto idx = extractKeyIndex(prefetchOp.getIndex());
        if (idx) {
          state.markPrefetched(*idx);
          LLVM_DEBUG(llvm::dbgs() << "  Marked as prefetched: ");
          LLVM_DEBUG(
              if (idx->isConstant()) {
                llvm::dbgs() << "constant " << *idx->constantValue << "\n";
              } else { llvm::dbgs() << "affine expression\n"; });
        }
        continue;
      }

      // Handle load_key operations
      if (auto loadOp = dyn_cast<LoadKeyOp>(&op)) {
        auto idx = extractKeyIndex(loadOp.getIndex());
        if (!idx) {
          loadOp.emitWarning("Could not extract key index for verification");
          continue;
        }

        if (!state.isPrefetched(*idx, ivValues)) {
          loadOp.emitError("load_key without corresponding prefetch_key");
          LLVM_DEBUG(llvm::dbgs() << "  Missing prefetch for: ");
          LLVM_DEBUG(
              if (idx->isConstant()) {
                llvm::dbgs() << "constant " << *idx->constantValue << "\n";
              } else { llvm::dbgs() << "affine expression\n"; });
          return failure();
        }

        LLVM_DEBUG(llvm::dbgs() << "  Verified load_key: ");
        LLVM_DEBUG(
            if (idx->isConstant()) {
              llvm::dbgs() << "constant " << *idx->constantValue << "\n";
            } else { llvm::dbgs() << "affine expression\n"; });
        continue;
      }

      // Handle affine.if operations
      if (auto ifOp = dyn_cast<affine::AffineIfOp>(&op)) {
        // Verify both branches
        VerificationState thenState = state;
        if (failed(verifyBlock(ifOp.getThenBlock(), thenState, ivValues))) {
          return failure();
        }

        VerificationState elseState = state;
        if (ifOp.hasElse()) {
          if (failed(verifyBlock(ifOp.getElseBlock(), elseState, ivValues))) {
            return failure();
          }
        }

        // Merge states from both branches
        state.merge(thenState);
        if (ifOp.hasElse()) {
          state.merge(elseState);
        }
        continue;
      }

      // Handle affine.for operations
      if (auto forOp = dyn_cast<affine::AffineForOp>(&op)) {
        if (failed(verifyAffineLoop(forOp, state, ivValues))) {
          return failure();
        }
        continue;
      }
    }

    return success();
  }

  // Verify an affine loop with abstract interpretation
  LogicalResult verifyAffineLoop(affine::AffineForOp forOp,
                                 VerificationState& state,
                                 DenseMap<Value, int64_t>& ivValues) {
    Value iv = forOp.getInductionVar();

    // Try to get constant loop bounds for concrete verification
    if (!forOp.hasConstantBounds()) {
      // For non-constant bounds, we do a symbolic verification
      // Just verify that the pattern looks correct
      LLVM_DEBUG(llvm::dbgs() << "  Verifying loop with symbolic bounds\n");

      // Create a copy of state for loop body
      VerificationState loopState = state;

      // Symbolically verify one iteration (with IV unknown)
      DenseMap<Value, int64_t> loopIvValues = ivValues;
      // We can't assign a specific value to IV, so we leave it out

      if (failed(verifyBlock(forOp.getBody(), loopState, loopIvValues))) {
        return failure();
      }

      // After the loop, we conservatively assume only pre-loop prefetches
      // remain (since in-loop prefetches are conditional)
      return success();
    }

    // For constant bounds, verify concrete iterations
    int64_t lb = forOp.getConstantLowerBound();
    int64_t ub = forOp.getConstantUpperBound();
    int64_t step = forOp.getStep().getSExtValue();

    LLVM_DEBUG(llvm::dbgs() << "  Verifying loop: " << lb << " to " << ub
                            << " step " << step << "\n");

    // Verify each iteration
    for (int64_t i = lb; i < ub; i += step) {
      VerificationState iterState = state;
      DenseMap<Value, int64_t> iterIvValues = ivValues;
      iterIvValues[iv] = i;

      LLVM_DEBUG(llvm::dbgs() << "    Iteration " << i << "\n");

      if (failed(verifyBlock(forOp.getBody(), iterState, iterIvValues))) {
        return failure();
      }

      // Update state for next iteration (prefetches from this iteration
      // are visible in next iteration)
      state = iterState;
    }

    return success();
  }

  // Main verification entry point
  LogicalResult verifyAllPrefetches(Operation* op) {
    VerificationState state;
    DenseMap<Value, int64_t> ivValues;

    // Walk all functions
    WalkResult result = op->walk([&](func::FuncOp funcOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Verifying function: " << funcOp.getName() << "\n");

      VerificationState funcState;
      DenseMap<Value, int64_t> funcIvValues;

      if (failed(verifyBlock(&funcOp.getBody().front(), funcState,
                             funcIvValues))) {
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Verification passed!\n");
    return success();
  }

  // ===================================================================
  // Runtime Delegated Prefetching
  // ===================================================================

  // Structure to track load_key operations and their context
  struct KeyLoadInfo {
    LoadKeyOp loadOp;
    affine::AffineForOp enclosingLoop = nullptr;  // Innermost enclosing loop
    SmallVector<affine::AffineForOp>
        loopNest;  // Full loop nest (outermost to innermost)
    SmallVector<ConditionalInfo>
        conditionalChain;  // Chain of enclosing affine.if (outermost to
                           // innermost)
    bool isInLoop() const { return enclosingLoop != nullptr; }
  };

  // Create a runtime-delegated prefetch list at the beginning of the program
  void createRuntimeDelegatedPrefetchList(Operation* moduleOp) {
    // Process each function
    moduleOp->walk([&](func::FuncOp funcOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Processing function: " << funcOp.getName() << "\n");

      // Collect all load_key operations with their context
      SmallVector<KeyLoadInfo> keyLoads;
      collectKeyLoads(funcOp, keyLoads);

      if (keyLoads.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "  No load_key operations found\n");
        return;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "  Found " << keyLoads.size() << " load_key operations\n");

      // Insert prefetch_key operations at the beginning of the function
      // preserving the execution order and loop structures
      OpBuilder builder(&funcOp.getBody().front().front());
      Location loc = funcOp.getLoc();

      insertPrefetchesAtBeginning(builder, loc, keyLoads);

      LLVM_DEBUG(llvm::dbgs() << "  Inserted runtime-delegated prefetches\n");
    });
  }

  // Collect all load_key operations from a function with their context
  void collectKeyLoads(func::FuncOp funcOp,
                       SmallVector<KeyLoadInfo>& keyLoads) {
    // Helper to collect from a block, tracking the loop nest and conditional
    // chain
    std::function<void(Block*, SmallVector<affine::AffineForOp>&,
                       SmallVector<ConditionalInfo>&)>
        collectFromBlock;
    collectFromBlock = [&](Block* block,
                           SmallVector<affine::AffineForOp>& loopNest,
                           SmallVector<ConditionalInfo>& conditionalChain) {
      for (Operation& op : *block) {
        if (auto loadOp = dyn_cast<LoadKeyOp>(&op)) {
          // llvm::errs() << "[KeyPrefetch-Runtime] Found load_key with " <<
          // conditionalChain.size() << " conditionals\n";
          KeyLoadInfo info;
          info.loadOp = loadOp;
          info.loopNest = loopNest;
          info.enclosingLoop = loopNest.empty() ? nullptr : loopNest.back();
          info.conditionalChain = conditionalChain;
          keyLoads.push_back(info);

        } else if (auto forOp = dyn_cast<affine::AffineForOp>(&op)) {
          // Recurse into loop
          loopNest.push_back(forOp);
          collectFromBlock(forOp.getBody(), loopNest, conditionalChain);
          loopNest.pop_back();

        } else if (auto ifOp = dyn_cast<affine::AffineIfOp>(&op)) {
          // Track the conditional chain for each branch
          SmallVector<ConditionalInfo> thenChain = conditionalChain;
          thenChain.push_back({ifOp, true});
          collectFromBlock(ifOp.getThenBlock(), loopNest, thenChain);

          if (ifOp.hasElse()) {
            SmallVector<ConditionalInfo> elseChain = conditionalChain;
            elseChain.push_back({ifOp, false});
            collectFromBlock(ifOp.getElseBlock(), loopNest, elseChain);
          }

        } else if (auto scfForOp = dyn_cast<scf::ForOp>(&op)) {
          // Handle SCF loops similarly
          // Note: Can't easily add to loopNest as it expects AffineForOp
          // For now, recurse but don't track
          collectFromBlock(scfForOp.getBody(), loopNest, conditionalChain);
        } else if (auto scfIfOp = dyn_cast<scf::IfOp>(&op)) {
          collectFromBlock(scfIfOp.thenBlock(), loopNest, conditionalChain);
          if (!scfIfOp.elseBlock()->empty()) {
            collectFromBlock(scfIfOp.elseBlock(), loopNest, conditionalChain);
          }
        }
      }
    };

    SmallVector<affine::AffineForOp> emptyNest;
    SmallVector<ConditionalInfo> emptyChain;
    collectFromBlock(&funcOp.getBody().front(), emptyNest, emptyChain);
  }

  // Insert all prefetch_key operations at the beginning, preserving order and
  // structure
  void insertPrefetchesAtBeginning(OpBuilder& builder, Location loc,
                                   SmallVectorImpl<KeyLoadInfo>& keyLoads) {
    // Track which loops we've already processed
    DenseSet<Operation*> processedLoops;

    // Process loads in order
    for (auto& info : keyLoads) {
      if (!info.isInLoop()) {
        // Sequential load: insert prefetch directly
        Value keyIndex = info.loadOp.getIndex();
        Value materializedIndex = rematerializeIndex(builder, loc, keyIndex);

        if (!materializedIndex) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  Warning: Failed to rematerialize index\n");
          continue;
        }

        // Cast to i64 if needed
        if (!materializedIndex.getType().isInteger(64)) {
          if (materializedIndex.getType().isIndex()) {
            materializedIndex = builder.create<arith::IndexCastOp>(
                loc, builder.getI64Type(), materializedIndex);
          } else if (materializedIndex.getType().isInteger()) {
            auto intType = cast<IntegerType>(materializedIndex.getType());
            if (intType.getWidth() < 64) {
              materializedIndex = builder.create<arith::ExtSIOp>(
                  loc, builder.getI64Type(), materializedIndex);
            } else if (intType.getWidth() > 64) {
              materializedIndex = builder.create<arith::TruncIOp>(
                  loc, builder.getI64Type(), materializedIndex);
            }
          }
        }

        builder.create<PrefetchKeyOp>(loc, materializedIndex);
        LLVM_DEBUG(llvm::dbgs() << "  Inserted sequential prefetch\n");

      } else {
        // Load inside loop: create prefetch loop structure
        affine::AffineForOp outermostLoop = info.loopNest.front();

        // Skip if already processed
        if (processedLoops.contains(outermostLoop.getOperation())) {
          continue;
        }
        processedLoops.insert(outermostLoop.getOperation());

        // Create loop structure that mirrors the original
        createPrefetchLoopStructure(builder, loc, outermostLoop);
      }
    }
  }

  // Create a loop structure for prefetching that mirrors the original loop
  void createPrefetchLoopStructure(OpBuilder& builder, Location loc,
                                   affine::AffineForOp originalLoop) {
    LLVM_DEBUG(llvm::dbgs() << "  Creating prefetch loop structure\n");

    // Clone the loop structure recursively with empty initial mapping
    IRMapping mapping;
    cloneLoopForPrefetch(builder, originalLoop, mapping);
  }

  // Clone a loop structure, replacing load_key with prefetch_key
  affine::AffineForOp cloneLoopForPrefetch(OpBuilder& builder,
                                           affine::AffineForOp originalLoop,
                                           IRMapping& mapping) {
    Location loc = originalLoop.getLoc();
    int64_t step = originalLoop.getStep().getSExtValue();

    // Map lower/upper bound operands if needed
    SmallVector<Value> lowerBoundOperands;
    for (Value operand : originalLoop.getLowerBoundOperands()) {
      lowerBoundOperands.push_back(mapping.lookupOrDefault(operand));
    }
    SmallVector<Value> upperBoundOperands;
    for (Value operand : originalLoop.getUpperBoundOperands()) {
      upperBoundOperands.push_back(mapping.lookupOrDefault(operand));
    }

    // Create the prefetch loop with same bounds
    auto prefetchLoop = builder.create<affine::AffineForOp>(
        loc, lowerBoundOperands, originalLoop.getLowerBoundMap(),
        upperBoundOperands, originalLoop.getUpperBoundMap(), step);

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(prefetchLoop.getBody());

    Value prefetchIV = prefetchLoop.getInductionVar();
    Value originalIV = originalLoop.getInductionVar();

    // Map the original IV to the prefetch IV
    mapping.map(originalIV, prefetchIV);

    // Process the loop body, looking for load_key and nested loops
    for (Operation& op : originalLoop.getBody()->without_terminator()) {
      if (auto loadOp = dyn_cast<LoadKeyOp>(&op)) {
        // Replace load_key with prefetch_key
        Value keyIndex = loadOp.getIndex();

        // Map the index through the current mapping
        Value mappedIndex = mapping.lookupOrDefault(keyIndex);
        if (mappedIndex == keyIndex) {
          // Not in mapping, try to compute it
          mappedIndex = computeIndexForLoopIV(builder, loc, originalIV,
                                              prefetchIV, keyIndex);
          if (!mappedIndex) {
            LLVM_DEBUG(llvm::dbgs()
                       << "  Warning: Failed to compute prefetch index\n");
            continue;
          }
        }

        // Cast to i64 if needed
        if (!mappedIndex.getType().isInteger(64)) {
          if (mappedIndex.getType().isIndex()) {
            mappedIndex = builder.create<arith::IndexCastOp>(
                loc, builder.getI64Type(), mappedIndex);
          } else if (mappedIndex.getType().isInteger()) {
            auto intType = cast<IntegerType>(mappedIndex.getType());
            if (intType.getWidth() < 64) {
              mappedIndex = builder.create<arith::ExtSIOp>(
                  loc, builder.getI64Type(), mappedIndex);
            } else if (intType.getWidth() > 64) {
              mappedIndex = builder.create<arith::TruncIOp>(
                  loc, builder.getI64Type(), mappedIndex);
            }
          }
        }

        builder.create<PrefetchKeyOp>(loc, mappedIndex);

      } else if (auto nestedForOp = dyn_cast<affine::AffineForOp>(&op)) {
        // Recursively clone nested loops with current mapping
        cloneLoopForPrefetch(builder, nestedForOp, mapping);

      } else if (auto ifOp = dyn_cast<affine::AffineIfOp>(&op)) {
        // Clone affine.if structure
        SmallVector<Value> mappedOperands;
        for (Value operand : ifOp.getOperands()) {
          mappedOperands.push_back(mapping.lookupOrDefault(operand));
        }

        auto prefetchIf = builder.create<affine::AffineIfOp>(
            loc, ifOp.getIntegerSet(), mappedOperands, ifOp.hasElse());

        // Clone then block
        {
          OpBuilder::InsertionGuard thenGuard(builder);
          builder.setInsertionPointToStart(prefetchIf.getThenBlock());
          cloneBlockForPrefetch(builder, ifOp.getThenBlock(), mapping);
        }

        // Clone else block if present
        if (ifOp.hasElse()) {
          OpBuilder::InsertionGuard elseGuard(builder);
          builder.setInsertionPointToStart(prefetchIf.getElseBlock());
          cloneBlockForPrefetch(builder, ifOp.getElseBlock(), mapping);
        }

      } else if (isa<arith::ConstantOp, arith::ConstantIndexOp>(&op)) {
        // Always clone constants
        Operation* cloned = builder.clone(op, mapping);

        // Update mapping with results
        for (auto [origResult, clonedResult] :
             llvm::zip(op.getResults(), cloned->getResults())) {
          mapping.map(origResult, clonedResult);
        }

      } else if (isa<arith::IndexCastOp, arith::ExtSIOp, arith::TruncIOp,
                     affine::AffineApplyOp>(&op)) {
        // For operations with operands, ensure all operands are available
        bool canClone = true;
        for (Value operand : op.getOperands()) {
          Value mappedOperand = mapping.lookupOrDefault(operand);
          if (mappedOperand == operand) {
            // Operand not in mapping - check if it's defined outside the loop
            if (auto defOp = operand.getDefiningOp()) {
              // Check if it's a constant we can rematerialize
              if (isa<arith::ConstantOp, arith::ConstantIndexOp>(defOp)) {
                // Rematerialize the constant
                Operation* rematerialized = builder.clone(*defOp);
                mapping.map(operand, rematerialized->getResult(0));
              } else if (!originalLoop->isAncestor(defOp)) {
                // Defined outside the loop and dominates - we can use it
                // Do nothing, it will be picked up by lookupOrDefault
              } else {
                // Defined inside the original loop but not yet in mapping
                canClone = false;
                break;
              }
            }
            // else: block argument from outside, should be available
          }
        }

        if (canClone) {
          Operation* cloned = builder.clone(op, mapping);

          // Update mapping with results
          for (auto [origResult, clonedResult] :
               llvm::zip(op.getResults(), cloned->getResults())) {
            mapping.map(origResult, clonedResult);
          }
        }
      }
      // Skip all other operations (FHE ops, clear_key, etc.)
    }

    // Update insertion point to after the loop
    builder.setInsertionPointAfter(prefetchLoop);
    return prefetchLoop;
  }

  // Clone a block for prefetching, replacing load_key with prefetch_key
  void cloneBlockForPrefetch(OpBuilder& builder, Block* block,
                             IRMapping& mapping) {
    for (Operation& op : block->without_terminator()) {
      if (auto loadOp = dyn_cast<LoadKeyOp>(&op)) {
        // Replace with prefetch_key
        Value keyIndex = loadOp.getIndex();
        Value mappedIndex = mapping.lookupOrDefault(keyIndex);

        // Cast to i64 if needed
        if (!mappedIndex.getType().isInteger(64)) {
          if (mappedIndex.getType().isIndex()) {
            mappedIndex = builder.create<arith::IndexCastOp>(
                loadOp.getLoc(), builder.getI64Type(), mappedIndex);
          }
        }

        builder.create<PrefetchKeyOp>(loadOp.getLoc(), mappedIndex);

      } else if (auto nestedForOp = dyn_cast<affine::AffineForOp>(&op)) {
        cloneLoopForPrefetch(builder, nestedForOp, mapping);

      } else if (auto ifOp = dyn_cast<affine::AffineIfOp>(&op)) {
        // Clone affine.if structure recursively
        SmallVector<Value> mappedOperands;
        for (Value operand : ifOp.getOperands()) {
          mappedOperands.push_back(mapping.lookupOrDefault(operand));
        }

        auto prefetchIf = builder.create<affine::AffineIfOp>(
            ifOp.getLoc(), ifOp.getIntegerSet(), mappedOperands,
            ifOp.hasElse());

        // Clone then block
        {
          OpBuilder::InsertionGuard thenGuard(builder);
          builder.setInsertionPointToStart(prefetchIf.getThenBlock());
          cloneBlockForPrefetch(builder, ifOp.getThenBlock(), mapping);
        }

        // Clone else block if present
        if (ifOp.hasElse()) {
          OpBuilder::InsertionGuard elseGuard(builder);
          builder.setInsertionPointToStart(prefetchIf.getElseBlock());
          cloneBlockForPrefetch(builder, ifOp.getElseBlock(), mapping);
        }

      } else if (isa<arith::ConstantOp, arith::ConstantIndexOp>(&op)) {
        // Always clone constants
        Operation* cloned = builder.clone(op, mapping);

        // Update mapping with results
        for (auto [origResult, clonedResult] :
             llvm::zip(op.getResults(), cloned->getResults())) {
          mapping.map(origResult, clonedResult);
        }

      } else if (isa<arith::IndexCastOp, arith::ExtSIOp, arith::TruncIOp,
                     affine::AffineApplyOp>(&op)) {
        // For operations with operands, ensure all operands are available
        for (Value operand : op.getOperands()) {
          Value mappedOperand = mapping.lookupOrDefault(operand);
          if (mappedOperand == operand) {
            // Operand not in mapping - check if it's a constant we can
            // rematerialize
            if (auto defOp = operand.getDefiningOp()) {
              if (isa<arith::ConstantOp, arith::ConstantIndexOp>(defOp)) {
                // Rematerialize the constant
                Operation* rematerialized = builder.clone(*defOp);
                mapping.map(operand, rematerialized->getResult(0));
              }
            }
          }
        }

        Operation* cloned = builder.clone(op, mapping);

        // Update mapping with results
        for (auto [origResult, clonedResult] :
             llvm::zip(op.getResults(), cloned->getResults())) {
          mapping.map(origResult, clonedResult);
        }
      }
    }
  }
};

}  // namespace

}  // namespace kmrt
}  // namespace heir
}  // namespace mlir
