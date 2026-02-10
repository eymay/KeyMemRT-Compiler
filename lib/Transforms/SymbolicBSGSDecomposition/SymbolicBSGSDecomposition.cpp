// File: lib/Transforms/SymbolicBSGSDecomposition/SymbolicBSGSDecomposition.cpp

#include "lib/Transforms/SymbolicBSGSDecomposition/SymbolicBSGSDecomposition.h"

#include <cmath>

#include "lib/Dialect/KMRT/IR/KMRTOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "llvm/include/llvm/Support/Debug.h"
#include "mlir/include/mlir/Dialect/Affine/Analysis/Utils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/LoopUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/IRMapping.h"              // from @llvm-project
#include "mlir/include/mlir/IR/IntegerSet.h"             // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SYMBOLICBSGSDECOMPOSITION
#include "lib/Transforms/SymbolicBSGSDecomposition/SymbolicBSGSDecomposition.h.inc"

// Helper to check if a value depends on the induction variable
static bool dependsOnIV(Value value, Value iv) {
  if (value == iv) return true;

  Operation *defOp = value.getDefiningOp();
  if (!defOp) return false;

  // Check through index_cast
  if (auto castOp = dyn_cast<arith::IndexCastOp>(defOp)) {
    return dependsOnIV(castOp.getIn(), iv);
  }

  // Check through affine.apply
  if (auto applyOp = dyn_cast<affine::AffineApplyOp>(defOp)) {
    for (Value operand : applyOp.getMapOperands()) {
      if (dependsOnIV(operand, iv)) return true;
    }
  }

  return false;
}

// Main pass implementation
struct SymbolicBSGSDecomposition
    : impl::SymbolicBSGSDecompositionBase<SymbolicBSGSDecomposition> {
  using SymbolicBSGSDecompositionBase::SymbolicBSGSDecompositionBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    llvm::dbgs() << "=== Symbolic BSGS Rotation Decomposition ===\n";

    // Walk through all affine.for loops
    func.walk([&](affine::AffineForOp loop) { processLoop(loop); });

    llvm::dbgs() << "Symbolic BSGS decomposition completed\n";
  }

 private:
  void processLoop(affine::AffineForOp loop) {
    // Get loop bounds
    if (!loop.hasConstantBounds()) {
      return;  // Skip non-constant bound loops
    }

    int64_t lowerBound = loop.getConstantLowerBound();
    int64_t upperBound = loop.getConstantUpperBound();
    int64_t rangeSize = upperBound - lowerBound;

    llvm::dbgs() << "Processing loop with bounds [" << lowerBound << ", " << upperBound << ")\n";

    Value iv = loop.getInductionVar();

    // Find rotation operations that depend on the IV
    SmallVector<openfhe::RotOp> rotationsToDecompose;

    loop.walk([&](openfhe::RotOp rotOp) {
      // Check if this rotation uses a dynamic key
      auto deserOp = rotOp.getEvalKey().getDefiningOp<kmrt::LoadKeyOp>();
      if (!deserOp) return;

      // Check if the rotation index depends on the IV
      Value rotIndex = deserOp.getIndex();
      if (dependsOnIV(rotIndex, iv)) {
        rotationsToDecompose.push_back(rotOp);
      }
    });

    if (rotationsToDecompose.empty()) {
      return;  // No rotations to decompose
    }

    llvm::dbgs() << "Found " << rotationsToDecompose.size()
                 << " rotations to decompose in loop with range " << rangeSize
                 << "\n";

    // Create three-loop BSGS structure
    OpBuilder builder(loop.getContext());
    createThreeLoopBSGS(builder, loop, rotationsToDecompose, rangeSize);
  }

  void createThreeLoopBSGS(OpBuilder &builder, affine::AffineForOp loop,
                           SmallVector<openfhe::RotOp> &rotationsToDecompose,
                           int64_t rangeSize) {
    Location loc = loop.getLoc();
    int64_t lowerBound = loop.getConstantLowerBound();
    int64_t upperBound = loop.getConstantUpperBound();

    // Get the original loop body operations
    SmallVector<Operation *> bodyOps;
    for (Operation &op : loop.getBody()->without_terminator()) {
      bodyOps.push_back(&op);
    }

    auto yieldOp = cast<affine::AffineYieldOp>(loop.getBody()->getTerminator());
    SmallVector<Value> yieldOperands(yieldOp.getOperands());

    builder.setInsertionPoint(loop);

    // Calculate baby step size (N2)
    int64_t defaultN2 = static_cast<int64_t>(std::ceil(std::sqrt(rangeSize)));
    Value N2Val = builder.create<arith::ConstantIndexOp>(loc, defaultN2);
    N2Val.getDefiningOp()->setAttr("bsgs.tunable_param",
                                   builder.getStringAttr("baby_step_size"));

    int64_t numFullGiants = rangeSize / defaultN2;
    int64_t remainderSize = rangeSize % defaultN2;

    llvm::dbgs() << "Creating nested loops with symbolic N2 (default = "
                 << defaultN2 << ", range = " << rangeSize
                 << ", full_giants = " << numFullGiants
                 << ", remainder = " << remainderSize << ")\n";

    // Allocate memref for baby step keys
    auto rotKeyType = kmrt::RotKeyType::get(builder.getContext(), std::nullopt);
    auto memrefType = MemRefType::get({defaultN2}, rotKeyType);
    Value keyMemref = builder.create<memref::AllocaOp>(loc, memrefType);

    SmallVector<Value> iterOperands(loop.getInits().begin(),
                                    loop.getInits().end());

    // ===================================================================
    // LOOP 1: PROLOGUE - First giant step (giant_step = 0)
    // Load baby step keys and compute with first giant step
    // ===================================================================
    auto prologueLoop = builder.create<affine::AffineForOp>(
        loc, 0, defaultN2, 1, iterOperands,
        [&](OpBuilder &proBuilder, Location proLoc, Value babyIV,
            ValueRange proIterArgs) {
          // Baby step index is just babyIV (giant step 0 * N2 + babyIV)
          Value actualIV = babyIV;

          // Load baby step key and store in memref
          auto loadedKey =
              proBuilder.create<kmrt::LoadKeyOp>(proLoc, rotKeyType, babyIV);
          proBuilder.create<memref::StoreOp>(proLoc, loadedKey.getRotKey(),
                                             keyMemref, ValueRange{babyIV});

          // Clone loop body with baby step rotation only (giant step is 0, no
          // rotation needed)
          IRMapping mapping;
          mapping.map(loop.getInductionVar(), actualIV);
          for (auto [oldArg, newArg] :
               llvm::zip(loop.getRegionIterArgs(), proIterArgs)) {
            mapping.map(oldArg, newArg);
          }

          cloneBodyWithBabyStepOnly(proBuilder, proLoc, bodyOps,
                                    rotationsToDecompose, mapping,
                                    yieldOperands, loadedKey.getRotKey(),
                                    std::nullopt, keyMemref, babyIV);
        });

    // Add attribute to prologue loop
    prologueLoop->setAttr("bsgs.loop_type", builder.getStringAttr("prologue"));

    // Track results for next loop
    SmallVector<Value> currentResults(prologueLoop.getResults().begin(),
                                      prologueLoop.getResults().end());

    // ===================================================================
    // LOOP 2: MAIN BSGS - Middle giant steps (giant_step = 1 to
    // numFullGiants-2) Use baby step keys from memref, no load/clear for baby
    // steps
    // ===================================================================
    if (numFullGiants > 2) {
      builder.setInsertionPointAfter(prologueLoop);

      auto mainLoop = builder.create<affine::AffineForOp>(
          loc, 1, numFullGiants - 1, 1, prologueLoop.getResults(),
          [&](OpBuilder &mainBuilder, Location mainLoc, Value giantIV,
              ValueRange mainIterArgs) {
            // Load giant step key
            auto giantAmtMap =
                AffineMap::get(1, 1,
                               mainBuilder.getAffineDimExpr(0) *
                                   mainBuilder.getAffineSymbolExpr(0),
                               mainBuilder.getContext());
            Value giantAmt = mainBuilder.create<affine::AffineApplyOp>(
                mainLoc, giantAmtMap, ValueRange{giantIV, N2Val});

            auto giantKey = mainBuilder.create<kmrt::LoadKeyOp>(
                mainLoc, rotKeyType, giantAmt);

            // Apply giant step rotations
            IRMapping giantStepMapping;
            for (openfhe::RotOp rotOp : rotationsToDecompose) {
              Value ctGiant = mainBuilder.create<openfhe::RotOp>(
                  mainLoc, rotOp.getType(), rotOp.getCryptoContext(),
                  rotOp.getCiphertext(), giantKey.getRotKey());
              giantStepMapping.map(rotOp.getResult(), ctGiant);
            }

            // Clear giant step key immediately
            mainBuilder.create<kmrt::ClearKeyOp>(mainLoc, giantKey.getRotKey());

            // Inner loop for baby steps - reuse keys from memref
            auto innerLoop = mainBuilder.create<affine::AffineForOp>(
                mainLoc, 0, defaultN2, 1, mainIterArgs,
                [&](OpBuilder &innerBuilder, Location innerLoc, Value babyIV,
                    ValueRange innerIterArgs) {
                  // Compute actual index
                  auto indexMap = AffineMap::get(
                      2, 1,
                      innerBuilder.getAffineDimExpr(0) *
                              innerBuilder.getAffineSymbolExpr(0) +
                          innerBuilder.getAffineDimExpr(1),
                      innerBuilder.getContext());
                  Value actualIV = innerBuilder.create<affine::AffineApplyOp>(
                      innerLoc, indexMap, ValueRange{giantIV, babyIV, N2Val});

                  // Load baby step key from memref and use it
                  Value keyFromMemref = innerBuilder.create<memref::LoadOp>(
                      innerLoc, keyMemref, ValueRange{babyIV});
                  auto useKeyOp = innerBuilder.create<kmrt::UseKeyOp>(
                      innerLoc, rotKeyType, keyFromMemref);

                  // Clone body with baby step rotation
                  IRMapping mapping;
                  mapping.map(loop.getInductionVar(), actualIV);
                  for (auto [oldArg, newArg] :
                       llvm::zip(loop.getRegionIterArgs(), innerIterArgs)) {
                    mapping.map(oldArg, newArg);
                  }

                  cloneBodyWithBabyStepOnly(
                      innerBuilder, innerLoc, bodyOps, rotationsToDecompose,
                      mapping, yieldOperands, useKeyOp.getRotKey(),
                      giantStepMapping, keyMemref, babyIV);
                });

            mainBuilder.create<affine::AffineYieldOp>(mainLoc,
                                                      innerLoop.getResults());
          });

      // Add attribute to main loop
      mainLoop->setAttr("bsgs.loop_type", builder.getStringAttr("main_bsgs"));

      // Update current results
      currentResults.clear();
      currentResults.append(mainLoop.getResults().begin(),
                            mainLoop.getResults().end());

      // Set insertion point after main loop for epilogue
      builder.setInsertionPointAfter(mainLoop);
    } else {
      // No main loop, insert epilogue after prologue
      builder.setInsertionPointAfter(prologueLoop);
    }

    // ===================================================================
    // LOOP 3: EPILOGUE - Last giant step (giant_step = numFullGiants-1)
    // Use baby step keys from memref and clear them after use
    // ===================================================================

    Value lastGiantIV =
        builder.create<arith::ConstantIndexOp>(loc, numFullGiants - 1);

    // Load giant step key for last iteration
    auto lastGiantAmtMap = AffineMap::get(
        1, 1, builder.getAffineDimExpr(0) * builder.getAffineSymbolExpr(0),
        builder.getContext());
    Value lastGiantAmt = builder.create<affine::AffineApplyOp>(
        loc, lastGiantAmtMap, ValueRange{lastGiantIV, N2Val});

    auto lastGiantKey =
        builder.create<kmrt::LoadKeyOp>(loc, rotKeyType, lastGiantAmt);

    // Apply giant step rotations
    IRMapping lastGiantMapping;
    for (openfhe::RotOp rotOp : rotationsToDecompose) {
      Value ctGiant = builder.create<openfhe::RotOp>(
          loc, rotOp.getType(), rotOp.getCryptoContext(), rotOp.getCiphertext(),
          lastGiantKey.getRotKey());
      lastGiantMapping.map(rotOp.getResult(), ctGiant);
    }

    // Clear giant step key
    builder.create<kmrt::ClearKeyOp>(loc, lastGiantKey.getRotKey());

    auto epilogueLoop = builder.create<affine::AffineForOp>(
        loc, 0, defaultN2, 1, currentResults,
        [&](OpBuilder &epiBuilder, Location epiLoc, Value babyIV,
            ValueRange epiIterArgs) {
          // Compute actual index
          auto indexMap =
              AffineMap::get(2, 1,
                             epiBuilder.getAffineDimExpr(0) *
                                     epiBuilder.getAffineSymbolExpr(0) +
                                 epiBuilder.getAffineDimExpr(1),
                             epiBuilder.getContext());
          Value actualIV = epiBuilder.create<affine::AffineApplyOp>(
              epiLoc, indexMap, ValueRange{lastGiantIV, babyIV, N2Val});

          // Load baby step key from memref and use it
          Value keyFromMemref = epiBuilder.create<memref::LoadOp>(
              epiLoc, keyMemref, ValueRange{babyIV});
          auto useKeyOp = epiBuilder.create<kmrt::UseKeyOp>(epiLoc, rotKeyType,
                                                            keyFromMemref);

          // Clone body with baby step rotation
          IRMapping mapping;
          mapping.map(loop.getInductionVar(), actualIV);
          for (auto [oldArg, newArg] :
               llvm::zip(loop.getRegionIterArgs(), epiIterArgs)) {
            mapping.map(oldArg, newArg);
          }

          cloneBodyWithBabyStepOnly(epiBuilder, epiLoc, bodyOps,
                                    rotationsToDecompose, mapping,
                                    yieldOperands, useKeyOp.getRotKey(),
                                    lastGiantMapping, keyMemref, babyIV,
                                    /*clearKeyAfter=*/keyFromMemref);
        });

    // Add attribute to epilogue loop
    epilogueLoop->setAttr("bsgs.loop_type", builder.getStringAttr("epilogue"));

    // Replace original loop
    loop.replaceAllUsesWith(epilogueLoop.getResults());
    loop.erase();
  }

  // Helper to clone loop body with baby step rotation only
  void cloneBodyWithBabyStepOnly(
      OpBuilder &builder, Location loc, SmallVector<Operation *> &bodyOps,
      SmallVector<openfhe::RotOp> &rotationsToDecompose, IRMapping &mapping,
      SmallVector<Value> &yieldOperands, Value babyKey,
      std::optional<IRMapping> giantStepMapping, Value keyMemref, Value babyIV,
      Value clearKeyAfter = Value()) {
    // Collect operations to skip
    SmallVector<Operation *> opsToSkip;
    for (openfhe::RotOp rotOp : rotationsToDecompose) {
      if (auto loadOp = rotOp.getEvalKey().getDefiningOp<kmrt::LoadKeyOp>()) {
        opsToSkip.push_back(loadOp);
        if (auto indexCastOp =
                loadOp.getIndex().getDefiningOp<arith::IndexCastOp>()) {
          opsToSkip.push_back(indexCastOp);
        }
      }
      for (Operation *user : rotOp.getEvalKey().getUsers()) {
        if (auto clearOp = dyn_cast<kmrt::ClearKeyOp>(user)) {
          opsToSkip.push_back(clearOp);
        }
      }
    }

    for (Operation *op : bodyOps) {
      if (llvm::find(opsToSkip, op) != opsToSkip.end()) {
        continue;
      }

      // Handle rotation operations
      if (auto rotOp = dyn_cast<openfhe::RotOp>(op)) {
        if (llvm::find(rotationsToDecompose, rotOp) !=
            rotationsToDecompose.end()) {
          Value ctInput;

          // Get giant-stepped ciphertext if available
          if (giantStepMapping.has_value()) {
            ctInput = giantStepMapping->lookup(rotOp.getResult());
          } else {
            // Prologue: no giant step, use original input
            ctInput = mapping.lookupOrDefault(rotOp.getCiphertext());
          }

          Value cryptoContext =
              mapping.lookupOrDefault(rotOp.getCryptoContext());

          // Apply baby step rotation
          Value ctBaby = builder.create<openfhe::RotOp>(
              loc, rotOp.getType(), cryptoContext, ctInput, babyKey);

          mapping.map(rotOp.getResult(), ctBaby);

          if (clearKeyAfter) {
            builder.create<kmrt::ClearKeyOp>(loc, clearKeyAfter);
          }
          continue;
        }
      }

      // Clone other operations normally
      Operation *clonedOp = builder.clone(*op, mapping);
      for (auto [oldResult, newResult] :
           llvm::zip(op->getResults(), clonedOp->getResults())) {
        mapping.map(oldResult, newResult);
      }
    }

    // Clear key before yield if requested (for epilogue loop)

    // Yield mapped results
    SmallVector<Value> mappedYields;
    for (Value yieldVal : yieldOperands) {
      mappedYields.push_back(mapping.lookup(yieldVal));
    }
    builder.create<affine::AffineYieldOp>(loc, mappedYields);
  }
};

}  // namespace heir
}  // namespace mlir
