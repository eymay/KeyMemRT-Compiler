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
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"

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

  // Try to evaluate to a constant given a map of induction variables to their values
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
        dimReplacements.push_back(getAffineConstantExpr(val, expr.getContext()));
      }
      SmallVector<AffineExpr> symReplacements;
      for (auto val : symValues) {
        symReplacements.push_back(getAffineConstantExpr(val, expr.getContext()));
      }

      auto constantExpr = expr.replaceDimsAndSymbols(dimReplacements, symReplacements);
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
      .Case<openfhe::RotOp, openfhe::AutomorphOp, RotationOp>(
          [](auto) { return 15; })
      .Case<openfhe::RelinOp>([](auto) { return 8; })
      .Case<openfhe::ModReduceOp, openfhe::LevelReduceOp>(
          [](auto) { return 3; })
      .Case<openfhe::KeySwitchOp>([](auto) { return 12; })
      // Key management operations (0 cost for prefetch calculation)
      .Case<LoadKeyOp, ClearKeyOp, PrefetchKeyOp>([](auto) { return 0; })
      // Default case
      .Default([](Operation*) { return 1; });
}

struct KMRTKeyPrefetching
    : impl::KMRTKeyPrefetchingBase<KMRTKeyPrefetching> {
  using impl::KMRTKeyPrefetchingBase<
      KMRTKeyPrefetching>::KMRTKeyPrefetchingBase;

  void runOnOperation() override {
    Operation* op = getOperation();

    // Find all load_key operations
    SmallVector<LoadKeyOp> loadKeyOps;
    op->walk([&](LoadKeyOp loadOp) { loadKeyOps.push_back(loadOp); });

    LLVM_DEBUG(llvm::dbgs() << "KMRTKeyPrefetching: Found " << loadKeyOps.size()
                            << " load_key operations\n");

    // Process each load_key operation
    for (auto loadOp : loadKeyOps) {
      // Check if this load is inside an affine loop
      if (auto parentLoop = loadOp->getParentOfType<affine::AffineForOp>()) {
        processAffineLoop(parentLoop);
      } else {
        insertPrefetchForLoad(loadOp);
      }
    }

    // Verify prefetches if requested
    if (verifyPrefetches.getValue()) {
      LLVM_DEBUG(llvm::dbgs() << "KMRTKeyPrefetching: Running verification\n");
      if (failed(verifyAllPrefetches(op))) {
        op->emitError("Key prefetch verification failed");
        signalPassFailure();
      }
    }
  }

 private:
  // Check if a prefetch for this index already exists nearby
  bool hasPrefetchNearby(Operation* startOp, Value indexValue, int lookback = 20) {
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

  // Insert prefetch for a single load_key operation (non-loop case)
  void insertPrefetchForLoad(LoadKeyOp loadOp) {
    Value indexValue = loadOp.getIndex();

    // Check if there's already a prefetch for this key nearby
    if (hasPrefetchNearby(loadOp.getOperation(), indexValue)) {
      LLVM_DEBUG(llvm::dbgs() << "KMRTKeyPrefetching: Skipping - prefetch "
                                 "already exists nearby\n");
      return;
    }

    // Walk backwards accumulating costs
    int64_t accumulatedCost = 0;
    Operation* current = loadOp.getOperation();
    Operation* placementPoint = current;

    while (current && accumulatedCost < prefetchThreshold.getValue()) {
      current = current->getPrevNode();
      if (!current) break;

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

    // Create the prefetch index value
    // If the load_key uses a constant, we need to materialize a new constant at the prefetch location
    // to avoid SSA dominance violations
    Value prefetchIndex;

    // Check if indexValue is a constant
    if (auto constOp = indexValue.getDefiningOp<arith::ConstantOp>()) {
      // Create a new constant at the prefetch location
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        int64_t constValue = intAttr.getInt();
        prefetchIndex = builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(constValue));
      } else {
        // Fallback: clone the constant
        prefetchIndex = builder.create<arith::ConstantOp>(loc, constOp.getValue());
      }
    } else if (auto constIndexOp = indexValue.getDefiningOp<arith::ConstantIndexOp>()) {
      // Create a new i64 constant from the index constant value
      int64_t constValue = constIndexOp.value();
      prefetchIndex = builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(constValue));
    } else {
      // Not a constant - use the original value (must be defined before placement point)
      prefetchIndex = indexValue;
    }

    // Ensure index is i64 for prefetch_key
    if (!prefetchIndex.getType().isInteger(64)) {
      if (prefetchIndex.getType().isIndex()) {
        prefetchIndex = builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), prefetchIndex);
      } else if (prefetchIndex.getType().isInteger()) {
        // Handle other integer types (e.g., i32, i16)
        auto intType = cast<IntegerType>(prefetchIndex.getType());
        if (intType.getWidth() < 64) {
          prefetchIndex = builder.create<arith::ExtSIOp>(loc, builder.getI64Type(), prefetchIndex);
        } else if (intType.getWidth() > 64) {
          prefetchIndex = builder.create<arith::TruncIOp>(loc, builder.getI64Type(), prefetchIndex);
        }
      }
    }

    builder.create<PrefetchKeyOp>(loc, prefetchIndex);

    LLVM_DEBUG(llvm::dbgs() << "KMRTKeyPrefetching: Inserted prefetch with "
                               "accumulated cost="
                            << accumulatedCost << "\n");
  }

  // Process an affine loop containing load_key operations
  void processAffineLoop(affine::AffineForOp forOp) {
    // Only process each loop once
    if (forOp->hasAttr("kmrt.prefetch_processed")) {
      return;
    }
    forOp->setAttr("kmrt.prefetch_processed",
                   UnitAttr::get(forOp.getContext()));

    // Find all load_key operations in this loop
    SmallVector<LoadKeyOp> loopLoadOps;
    forOp.walk([&](LoadKeyOp loadOp) {
      // Only process loads directly in this loop (not nested loops)
      if (loadOp->getParentOfType<affine::AffineForOp>() == forOp) {
        loopLoadOps.push_back(loadOp);
      }
    });

    if (loopLoadOps.empty()) {
      return;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "KMRTKeyPrefetching: Processing affine loop with "
               << loopLoadOps.size() << " load_key operations\n");

    // For each load_key in the loop, analyze cost from loop start to the load
    for (auto loadOp : loopLoadOps) {
      processLoopLoadKey(forOp, loadOp);
    }
  }

  // Process a single load_key operation within an affine loop
  void processLoopLoadKey(affine::AffineForOp forOp, LoadKeyOp loadOp) {
    Operation* loopBodyStart = &forOp.getBody()->front();

    // Calculate total cost of the entire loop body (one iteration)
    int64_t totalLoopCost = 0;
    Operation* current = loopBodyStart;
    while (current) {
      int64_t opCost = getOperationCost(current);
      if (opCost > 0) {
        totalLoopCost += opCost;
      }
      current = current->getNextNode();
    }

    LLVM_DEBUG(llvm::dbgs() << "KMRTKeyPrefetching: Total loop body cost = "
                            << totalLoopCost << ", threshold = "
                            << prefetchThreshold.getValue() << "\n");

    // Calculate prefetch distance in iterations
    // This is the sweet spot: how many iterations ahead should we prefetch?
    // If loop cost is 20 and threshold is 50, we prefetch ~2 iterations ahead
    unsigned prefetchDistance = 1;  // Default minimum
    if (totalLoopCost > 0) {
      prefetchDistance = std::max(1U,
          static_cast<unsigned>(prefetchThreshold.getValue() / totalLoopCost));
    }

    // Calculate how many iterations we can prefetch before the loop
    // Get loop bounds
    int64_t lowerBound = forOp.getConstantLowerBound();
    int64_t upperBound = forOp.getConstantUpperBound();
    int64_t totalIterations = upperBound - lowerBound;

    // Pre-loop prefetches: we can prefetch up to prefetchDistance iterations
    unsigned preLoopIterations = std::min(prefetchDistance,
                                          static_cast<unsigned>(totalIterations));

    // For in-loop prefetching, we place it at the beginning of the loop body
    // and prefetch for iteration (i + prefetchDistance)
    Operation* prefetchPlacement = loopBodyStart;

    LLVM_DEBUG(llvm::dbgs() << "KMRTKeyPrefetching: Prefetch distance = "
                            << prefetchDistance << " iterations"
                            << ", Pre-loop iterations = " << preLoopIterations << "\n");

    // Insert prefetches before the loop
    insertPreLoopPrefetches(forOp, loadOp, preLoopIterations);

    // Insert in-loop prefetch at the beginning of loop body
    insertInLoopPrefetch(forOp, loadOp, prefetchPlacement, prefetchDistance);
  }

  // Insert prefetches before the loop for the first N iterations
  void insertPreLoopPrefetches(affine::AffineForOp forOp, LoadKeyOp loadOp,
                                unsigned numIterations) {
    if (numIterations == 0) return;

    // Get loop bounds
    int64_t lowerBound = forOp.getConstantLowerBound();
    int64_t upperBound = forOp.getConstantUpperBound();

    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();
    Value iv = forOp.getInductionVar();

    // Analyze the load's index to understand the pattern
    Value loadIndex = loadOp.getIndex();

    // For each iteration we want to prefetch before the loop
    for (unsigned i = 0; i < numIterations; i++) {
      int64_t iterValue = lowerBound + i;
      if (iterValue >= upperBound) break;

      // Compute the key index for this iteration using the same pattern as the load
      Value prefetchIndex = computeIndexForIteration(builder, loc, iv, loadIndex, iterValue);

      // Ensure it's i64 for prefetch_key
      if (prefetchIndex) {
        // Cast to i64 if it's not already i64
        if (!prefetchIndex.getType().isInteger(64)) {
          if (prefetchIndex.getType().isIndex()) {
            prefetchIndex = builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), prefetchIndex);
          } else if (prefetchIndex.getType().isInteger()) {
            // Handle other integer types (e.g., i32, i16)
            auto intType = cast<IntegerType>(prefetchIndex.getType());
            if (intType.getWidth() < 64) {
              prefetchIndex = builder.create<arith::ExtSIOp>(loc, builder.getI64Type(), prefetchIndex);
            } else if (intType.getWidth() > 64) {
              prefetchIndex = builder.create<arith::TruncIOp>(loc, builder.getI64Type(), prefetchIndex);
            }
          }
        }

        // Check if prefetch already exists
        if (!hasPrefetchNearby(forOp.getOperation(), prefetchIndex, 30)) {
          builder.create<PrefetchKeyOp>(loc, prefetchIndex);

          LLVM_DEBUG(llvm::dbgs()
                     << "KMRTKeyPrefetching: Inserted pre-loop prefetch for iteration "
                     << iterValue << "\n");
        }
      }
    }
  }

  // Compute the key index for a specific iteration value
  // Given a load index pattern and a concrete iteration value, compute what index it represents
  Value computeIndexForIteration(OpBuilder& builder, Location loc, Value loopIV,
                                  Value loadIndex, int64_t iterValue) {
    // Try to trace back through index_cast
    if (auto castOp = loadIndex.getDefiningOp<arith::IndexCastOp>()) {
      Value innerIndex = computeIndexForIteration(builder, loc, loopIV, castOp.getIn(), iterValue);
      if (!innerIndex) return nullptr;

      // Apply the same cast only if needed (if innerIndex is not already the target type)
      Type targetType = castOp.getType();
      if (innerIndex.getType() != targetType) {
        return builder.create<arith::IndexCastOp>(loc, targetType, innerIndex);
      }
      return innerIndex;
    }

    // If it's an affine.apply, we need to apply the same map with the concrete value
    if (auto applyOp = loadIndex.getDefiningOp<affine::AffineApplyOp>()) {
      AffineMap map = applyOp.getAffineMap();
      SmallVector<Value> operands;

      // Replace loop IV with concrete iteration value, keep other operands
      for (Value operand : applyOp.getOperands()) {
        if (operand == loopIV) {
          // Create constant for this iteration
          operands.push_back(builder.create<arith::ConstantIndexOp>(loc, iterValue));
        } else {
          operands.push_back(operand);
        }
      }

      // Apply the affine map with substituted operands
      Value result = builder.create<affine::AffineApplyOp>(loc, map, operands);

      // Try to constant fold if all operands are constants
      // This is important for generating clean IR
      bool allConst = true;
      SmallVector<int64_t> constOperands;
      for (Value operand : operands) {
        if (auto constOp = operand.getDefiningOp<arith::ConstantOp>()) {
          if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
            constOperands.push_back(intAttr.getInt());
          } else {
            allConst = false;
            break;
          }
        } else if (auto constIndexOp = operand.getDefiningOp<arith::ConstantIndexOp>()) {
          constOperands.push_back(constIndexOp.value());
        } else {
          allConst = false;
          break;
        }
      }

      if (allConst && map.getNumResults() == 1) {
        // Evaluate the map
        SmallVector<int64_t> dims, syms;
        unsigned numDims = map.getNumDims();
        for (unsigned i = 0; i < numDims; i++) {
          dims.push_back(constOperands[i]);
        }
        for (unsigned i = numDims; i < constOperands.size(); i++) {
          syms.push_back(constOperands[i]);
        }

        // Replace dims and symbols with constants
        SmallVector<AffineExpr> dimReplacements, symReplacements;
        for (int64_t val : dims) {
          dimReplacements.push_back(getAffineConstantExpr(val, builder.getContext()));
        }
        for (int64_t val : syms) {
          symReplacements.push_back(getAffineConstantExpr(val, builder.getContext()));
        }

        auto expr = map.getResult(0).replaceDimsAndSymbols(dimReplacements, symReplacements);
        if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
          // Create a constant index (will be cast to i64 later if needed)
          return builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(constExpr.getValue()));
        }
      }

      return result;
    }

    // If it's the loop IV directly, just return the iteration value as i64
    if (loadIndex == loopIV) {
      return builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(iterValue));
    }

    // If it's a constant, we need to materialize a new constant to avoid dominance violations
    if (auto constOp = loadIndex.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        int64_t constValue = intAttr.getInt();
        return builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(constValue));
      } else {
        // Fallback: clone the constant
        return builder.create<arith::ConstantOp>(loc, constOp.getValue());
      }
    } else if (auto constIndexOp = loadIndex.getDefiningOp<arith::ConstantIndexOp>()) {
      int64_t constValue = constIndexOp.value();
      return builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(constValue));
    }

    // If it's an SSA value not derived from IV and not a constant, use it directly
    // (This should be rare - the value must dominate the prefetch location)
    return loadIndex;
  }

  // Compute the key index for a future iteration (for in-loop prefetching)
  // Given the load index pattern and a Value representing the next iteration, compute the index
  Value computeIndexForNextIteration(OpBuilder& builder, Location loc, Value loopIV,
                                      Value loadIndex, Value nextIterValue) {
    // Try to trace back through index_cast
    if (auto castOp = loadIndex.getDefiningOp<arith::IndexCastOp>()) {
      Value innerIndex = computeIndexForNextIteration(builder, loc, loopIV, castOp.getIn(), nextIterValue);
      if (!innerIndex) return nullptr;

      // Apply the same cast only if needed
      Type targetType = castOp.getType();
      if (innerIndex.getType() != targetType) {
        return builder.create<arith::IndexCastOp>(loc, targetType, innerIndex);
      }
      return innerIndex;
    }

    // If it's an affine.apply, we need to apply the same map with the next iteration value
    if (auto applyOp = loadIndex.getDefiningOp<affine::AffineApplyOp>()) {
      AffineMap map = applyOp.getAffineMap();
      SmallVector<Value> operands;

      // Replace loop IV with next iteration value, keep other operands
      for (Value operand : applyOp.getOperands()) {
        if (operand == loopIV) {
          operands.push_back(nextIterValue);
        } else {
          operands.push_back(operand);
        }
      }

      // Apply the affine map with substituted operands
      return builder.create<affine::AffineApplyOp>(loc, map, operands);
    }

    // If it's the loop IV directly, just return the next iteration value
    if (loadIndex == loopIV) {
      return nextIterValue;
    }

    // If it's a constant, we need to materialize a new constant to avoid dominance violations
    // For in-loop prefetches, the constant will be created inside the affine.if block
    if (auto constOp = loadIndex.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        int64_t constValue = intAttr.getInt();
        return builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(constValue));
      } else {
        // Fallback: clone the constant
        return builder.create<arith::ConstantOp>(loc, constOp.getValue());
      }
    } else if (auto constIndexOp = loadIndex.getDefiningOp<arith::ConstantIndexOp>()) {
      int64_t constValue = constIndexOp.value();
      return builder.create<arith::ConstantOp>(loc, builder.getI64IntegerAttr(constValue));
    }

    // If it's an SSA value not derived from IV and not a constant, use it directly
    // (This value should dominate the loop, so it's safe to use)
    return loadIndex;
  }

  // Insert prefetch inside the loop at a specific placement point
  void insertInLoopPrefetch(affine::AffineForOp forOp, LoadKeyOp loadOp,
                             Operation* placementPoint, unsigned prefetchDistance) {
    OpBuilder builder(placementPoint);
    Location loc = forOp.getLoc();
    MLIRContext* ctx = forOp.getContext();

    // Get loop induction variable
    Value iv = forOp.getInductionVar();

    // For the affine.if condition: d0 + prefetchDistance < upperBound
    // We need to handle both constant and symbolic upper bounds

    // Check if the loop has a constant upper bound
    if (forOp.hasConstantUpperBound()) {
      int64_t upperBound = forOp.getConstantUpperBound();

      // Create affine condition: d0 + prefetchDistance < upperBound
      // In >= 0 form: upperBound - prefetchDistance - d0 - 1 >= 0
      // Which simplifies to: -d0 + (upperBound - prefetchDistance - 1) >= 0
      auto d0 = getAffineDimExpr(0, ctx);
      auto constraint = -d0 + (upperBound - prefetchDistance - 1);

      SmallVector<bool> eqFlags{false};  // Inequality (>= 0)
      auto affineSet = IntegerSet::get(1, 0, {constraint}, eqFlags);

      // Create affine.if condition
      auto affineIf = builder.create<affine::AffineIfOp>(loc, affineSet,
                                                          ValueRange{iv}, false);

      // Inside the affine.if, compute the key index for (iv + prefetchDistance)
      builder.setInsertionPointToStart(affineIf.getThenBlock());

      // Create affine map: (d0) -> (d0 + prefetchDistance)
      auto map = AffineMap::get(1, 0, d0 + prefetchDistance, ctx);
      Value nextIter =
          builder.create<affine::AffineApplyOp>(loc, map, ValueRange{iv});

      // Compute the key index for the next iteration using the same pattern as the load
      Value prefetchIndex = computeIndexForNextIteration(builder, loc, iv, loadOp.getIndex(), nextIter);

      // Ensure it's i64 for prefetch_key
      if (prefetchIndex) {
        // Cast to i64 if it's not already i64
        if (!prefetchIndex.getType().isInteger(64)) {
          if (prefetchIndex.getType().isIndex()) {
            prefetchIndex = builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), prefetchIndex);
          } else if (prefetchIndex.getType().isInteger()) {
            // Handle other integer types (e.g., i32, i16)
            auto intType = cast<IntegerType>(prefetchIndex.getType());
            if (intType.getWidth() < 64) {
              prefetchIndex = builder.create<arith::ExtSIOp>(loc, builder.getI64Type(), prefetchIndex);
            } else if (intType.getWidth() > 64) {
              prefetchIndex = builder.create<arith::TruncIOp>(loc, builder.getI64Type(), prefetchIndex);
            }
          }
        }

        // Insert prefetch
        builder.create<PrefetchKeyOp>(loc, prefetchIndex);
      }

      LLVM_DEBUG(llvm::dbgs() << "KMRTKeyPrefetching: Inserted in-loop prefetch "
                                 "at distance "
                              << prefetchDistance << " iterations\n");
    } else {
      // Symbolic/non-constant upper bound
      // Get the upper bound map and its operands
      AffineMap ubMap = forOp.getUpperBoundMap();
      SmallVector<Value> ubOperands(forOp.getUpperBoundOperands().begin(),
                                     forOp.getUpperBoundOperands().end());

      // The upper bound map can have dimensions and/or symbols
      // For affine.if, we need to remap them:
      // - Loop IV becomes d0 in affine.if
      // - Upper bound map dimensions become d1, d2, ... in affine.if
      // - Upper bound map symbols become s0, s1, ... in affine.if

      unsigned numUBDims = ubMap.getNumDims();
      unsigned numUBSyms = ubMap.getNumSymbols();

      // Remap upper bound expression to affine.if context
      // d0 in affine.if = loop IV
      // d1, d2, ... in affine.if = dimensions from ubMap
      // s0, s1, ... in affine.if = symbols from ubMap
      auto d0 = getAffineDimExpr(0, ctx);  // Loop IV

      // Remap the upper bound expression
      SmallVector<AffineExpr> dimReplacements;
      for (unsigned i = 0; i < numUBDims; i++) {
        dimReplacements.push_back(getAffineDimExpr(i + 1, ctx));
      }
      SmallVector<AffineExpr> symReplacements;
      for (unsigned i = 0; i < numUBSyms; i++) {
        symReplacements.push_back(getAffineSymbolExpr(i, ctx));
      }

      AffineExpr ubExpr = ubMap.getResult(0);
      AffineExpr remappedUB = ubExpr.replaceDimsAndSymbols(dimReplacements, symReplacements);

      // Create constraint: d0 + prefetchDistance < remappedUB
      // In >= 0 form: remappedUB - prefetchDistance - d0 - 1 >= 0
      auto constraint = remappedUB - getAffineConstantExpr(prefetchDistance + 1, ctx) - d0;

      SmallVector<bool> eqFlags{false};  // Inequality (>= 0)

      // The affine set has (1 + numUBDims) dimensions and numUBSyms symbols
      auto affineSet = IntegerSet::get(1 + numUBDims, numUBSyms, {constraint}, eqFlags);

      // Operands for affine.if: d0 (loop IV), d1..dN (ubMap dims), s0..sM (ubMap syms)
      SmallVector<Value> affineIfOperands;
      affineIfOperands.push_back(iv);
      affineIfOperands.append(ubOperands.begin(), ubOperands.end());

      // Create affine.if condition
      auto affineIf = builder.create<affine::AffineIfOp>(loc, affineSet,
                                                          affineIfOperands, false);

      // Inside the affine.if, compute the key index for (iv + prefetchDistance)
      builder.setInsertionPointToStart(affineIf.getThenBlock());

      // Create affine map: (d0) -> (d0 + prefetchDistance)
      auto map = AffineMap::get(1, 0, d0 + prefetchDistance, ctx);
      Value nextIter =
          builder.create<affine::AffineApplyOp>(loc, map, ValueRange{iv});

      // Compute the key index for the next iteration using the same pattern as the load
      Value prefetchIndex = computeIndexForNextIteration(builder, loc, iv, loadOp.getIndex(), nextIter);

      // Ensure it's i64 for prefetch_key
      if (prefetchIndex) {
        // Cast to i64 if it's not already i64
        if (!prefetchIndex.getType().isInteger(64)) {
          if (prefetchIndex.getType().isIndex()) {
            prefetchIndex = builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), prefetchIndex);
          } else if (prefetchIndex.getType().isInteger()) {
            // Handle other integer types (e.g., i32, i16)
            auto intType = cast<IntegerType>(prefetchIndex.getType());
            if (intType.getWidth() < 64) {
              prefetchIndex = builder.create<arith::ExtSIOp>(loc, builder.getI64Type(), prefetchIndex);
            } else if (intType.getWidth() > 64) {
              prefetchIndex = builder.create<arith::TruncIOp>(loc, builder.getI64Type(), prefetchIndex);
            }
          }
        }

        // Insert prefetch
        builder.create<PrefetchKeyOp>(loc, prefetchIndex);
      }

      LLVM_DEBUG(llvm::dbgs() << "KMRTKeyPrefetching: Inserted in-loop prefetch "
                                 "at distance "
                              << prefetchDistance << " iterations (symbolic bound)\n");
    }
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
          LLVM_DEBUG(if (idx->isConstant()) {
            llvm::dbgs() << "constant " << *idx->constantValue << "\n";
          } else {
            llvm::dbgs() << "affine expression\n";
          });
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
          LLVM_DEBUG(if (idx->isConstant()) {
            llvm::dbgs() << "constant " << *idx->constantValue << "\n";
          } else {
            llvm::dbgs() << "affine expression\n";
          });
          return failure();
        }

        LLVM_DEBUG(llvm::dbgs() << "  Verified load_key: ");
        LLVM_DEBUG(if (idx->isConstant()) {
          llvm::dbgs() << "constant " << *idx->constantValue << "\n";
        } else {
          llvm::dbgs() << "affine expression\n";
        });
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

      // After the loop, we conservatively assume only pre-loop prefetches remain
      // (since in-loop prefetches are conditional)
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
      LLVM_DEBUG(llvm::dbgs() << "Verifying function: " << funcOp.getName()
                              << "\n");

      VerificationState funcState;
      DenseMap<Value, int64_t> funcIvValues;

      if (failed(verifyBlock(&funcOp.getBody().front(), funcState, funcIvValues))) {
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
};

}  // namespace

}  // namespace kmrt
}  // namespace heir
}  // namespace mlir
