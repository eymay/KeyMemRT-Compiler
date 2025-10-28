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
#include "mlir/include/mlir/IR/IRMapping.h"     // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"  // from @llvm-project
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

    // Create nested loops for BSGS decomposition
    OpBuilder builder(loop.getContext());
    createNestedBSGSLoops(builder, loop, rotationsToDecompose, rangeSize);
  }

  void createNestedBSGSLoops(OpBuilder &builder, affine::AffineForOp loop,
                             SmallVector<openfhe::RotOp> &rotationsToDecompose,
                             int64_t rangeSize) {
    Location loc = loop.getLoc();
    int64_t lowerBound = loop.getConstantLowerBound();
    int64_t upperBound = loop.getConstantUpperBound();

    // Get the original loop body operations (we'll need to clone them)
    SmallVector<Operation *> bodyOps;
    for (Operation &op : loop.getBody()->without_terminator()) {
      bodyOps.push_back(&op);
    }

    // Get loop results info
    auto yieldOp = cast<affine::AffineYieldOp>(loop.getBody()->getTerminator());
    SmallVector<Value> yieldOperands(yieldOp.getOperands());

    builder.setInsertionPoint(loop);

    // Create symbolic N2 parameter (baby step size)
    // This is left as a runtime value that can be tuned by a later optimization
    // pass Default heuristic: ceil(sqrt(rangeSize))
    int64_t defaultN2 = static_cast<int64_t>(std::ceil(std::sqrt(rangeSize)));
    Value N2Val = builder.create<arith::ConstantIndexOp>(loc, defaultN2);

    // Add an attribute to mark this as a tunable parameter
    N2Val.getDefiningOp()->setAttr("bsgs.tunable_param",
                                   builder.getStringAttr("baby_step_size"));

    // Calculate number of full giant step iterations and remainder
    int64_t numFullGiants = rangeSize / defaultN2;
    int64_t remainderSize = rangeSize % defaultN2;

    llvm::dbgs() << "Creating nested loops with symbolic N2 (default = "
                 << defaultN2 << ", range = " << rangeSize
                 << ", full_giants = " << numFullGiants
                 << ", remainder = " << remainderSize << ")\n";

    // Create outer loop for giant steps (full iterations only)
    // Upper bound: rangeSize / N2 (integer division)
    auto outerUBMap = builder.getConstantAffineMap(numFullGiants);

    SmallVector<Value> iterOperands(loop.getInits().begin(),
                                    loop.getInits().end());
    auto outerLoop = builder.create<affine::AffineForOp>(
        loc, /*lbOperands=*/ValueRange{},
        /*lbMap=*/builder.getConstantAffineMap(0),
        /*ubOperands=*/ValueRange{},
        /*ubMap=*/outerUBMap,
        /*step=*/1, iterOperands,
        [&](OpBuilder &outerBuilder, Location outerLoc, Value giantIV,
            ValueRange outerIterArgs) {
          // === GIANT STEP OPERATIONS IN OUTER LOOP ===
          // These happen once per outer loop iteration, before the inner loop

          // We need to process rotations and create giant step operations
          // We'll store the giant-stepped ciphertexts and pass them to inner
          // loop
          IRMapping giantStepMapping;

          // Map the crypto context and input ciphertext for rotations
          for (openfhe::RotOp rotOp : rotationsToDecompose) {
            Value inputCt = rotOp.getCiphertext();
            Value cryptoContext = rotOp.getCryptoContext();

            // Giant step amount: giantIV * N2
            auto giantAmtMap =
                AffineMap::get(1, 1,
                               outerBuilder.getAffineDimExpr(0) *
                                   outerBuilder.getAffineSymbolExpr(0),
                               outerBuilder.getContext());
            SmallVector<Value> giantAmtOperands{giantIV, N2Val};
            Value giantAmt = outerBuilder.create<affine::AffineApplyOp>(
                outerLoc, giantAmtMap, giantAmtOperands);

            // Load key for giant step - dynamic rotation
            // (giantAmt is computed from giantIV * N2, runtime-determined)
            auto rotKeyType = kmrt::RotKeyType::get(outerBuilder.getContext(), std::nullopt);
            Value rkGiant = outerBuilder.create<kmrt::LoadKeyOp>(
                outerLoc, rotKeyType, giantAmt);

            // Use KMRT RotKey directly with OpenFHE rotation
            Value ctGiant = outerBuilder.create<openfhe::RotOp>(
                outerLoc, rotOp.getType(), cryptoContext, inputCt, rkGiant);

            // KMRT clear
            outerBuilder.create<kmrt::ClearKeyOp>(outerLoc, rkGiant);

            // Store the giant-stepped ciphertext for use in inner loop
            giantStepMapping.map(rotOp.getResult(), ctGiant);
          }

          // Create inner loop for baby steps (0 to N2)
          // For full giant step iterations, inner loop goes from 0 to N2
          auto innerLoop = outerBuilder.create<affine::AffineForOp>(
              outerLoc, /*lbOperands=*/ValueRange{},
              /*lbMap=*/outerBuilder.getConstantAffineMap(0),
              /*ubOperands=*/ValueRange{},
              /*ubMap=*/outerBuilder.getConstantAffineMap(defaultN2),
              /*step=*/1, outerIterArgs,
              [&](OpBuilder &innerBuilder, Location innerLoc, Value babyIV,
                  ValueRange innerIterArgs) {
                // Compute actual iteration index: i = giantIV * N2 + babyIV
                // Using symbolic N2 as a symbol in the affine map
                auto indexMap =
                    AffineMap::get(2, 1,
                                   innerBuilder.getAffineDimExpr(0) *
                                           innerBuilder.getAffineSymbolExpr(0) +
                                       innerBuilder.getAffineDimExpr(1),
                                   innerBuilder.getContext());
                SmallVector<Value> indexOperands{giantIV, babyIV, N2Val};
                Value actualIV = innerBuilder.create<affine::AffineApplyOp>(
                    innerLoc, indexMap, indexOperands);

                // Clone the loop body, replacing rotations with baby step only
                IRMapping mapping;
                mapping.map(loop.getInductionVar(), actualIV);
                for (auto [oldArg, newArg] :
                     llvm::zip(loop.getRegionIterArgs(), innerIterArgs)) {
                  mapping.map(oldArg, newArg);
                }

                // Collect operations to skip (load_key/clear_key associated
                // with decomposed rotations)
                SmallVector<Operation *> opsToSkip;
                for (openfhe::RotOp rotOp : rotationsToDecompose) {
                  // Skip the KMRT load_key operation
                  if (auto loadOp =
                          rotOp.getEvalKey().getDefiningOp<kmrt::LoadKeyOp>()) {
                    opsToSkip.push_back(loadOp);
                    // Also skip the index_cast that feeds into the load_key
                    if (auto indexCastOp =
                            loadOp.getIndex()
                                .getDefiningOp<arith::IndexCastOp>()) {
                      opsToSkip.push_back(indexCastOp);
                    }
                  }
                  // Find clear_key ops that use this rotation's eval key
                  for (Operation *user : rotOp.getEvalKey().getUsers()) {
                    if (auto clearOp = dyn_cast<kmrt::ClearKeyOp>(user)) {
                      opsToSkip.push_back(clearOp);
                    }
                  }
                }

                SmallVector<Value> results;
                for (Operation *op : bodyOps) {
                  // Skip operations associated with rotations we're decomposing
                  if (llvm::find(opsToSkip, op) != opsToSkip.end()) {
                    continue;
                  }

                  // Check if this is a rotation to decompose
                  if (auto rotOp = dyn_cast<openfhe::RotOp>(op)) {
                    if (llvm::find(rotationsToDecompose, rotOp) !=
                        rotationsToDecompose.end()) {
                      // Apply only baby step rotation (giant step was done in
                      // outer loop)
                      Value ctGiant =
                          giantStepMapping.lookup(rotOp.getResult());
                      Value result = applyBabyStepRotation(
                          innerBuilder, rotOp, babyIV, ctGiant, mapping);
                      mapping.map(rotOp.getResult(), result);
                      continue;
                    }
                  }

                  // Clone the operation normally
                  Operation *clonedOp = innerBuilder.clone(*op, mapping);
                  for (auto [oldResult, newResult] :
                       llvm::zip(op->getResults(), clonedOp->getResults())) {
                    mapping.map(oldResult, newResult);
                  }
                }

                // Yield the mapped results
                SmallVector<Value> mappedYields;
                for (Value yieldVal : yieldOperands) {
                  mappedYields.push_back(mapping.lookup(yieldVal));
                }
                innerBuilder.create<affine::AffineYieldOp>(innerLoc,
                                                           mappedYields);
              });

          // Outer loop yields the results from inner loop
          outerBuilder.create<affine::AffineYieldOp>(outerLoc,
                                                     innerLoop.getResults());
        });

    // Handle remainder iterations if any
    // The remainder is just a single giant step with partial baby steps
    Operation *finalLoop;
    if (remainderSize > 0) {
      builder.setInsertionPointAfter(outerLoop);

      // Giant step offset for remainder: numFullGiants * defaultN2
      int64_t giantOffset = numFullGiants * defaultN2;

      // Apply the giant step rotation (just once, not in a loop)
      IRMapping giantStepMapping;
      for (openfhe::RotOp rotOp : rotationsToDecompose) {
        Value inputCt = rotOp.getCiphertext();
        Value cryptoContext = rotOp.getCryptoContext();

        // Giant step amount: constant offset
        Value giantAmt = builder.create<arith::ConstantIndexOp>(loc, giantOffset);

        // Load key for giant step
        auto rotKeyType = kmrt::RotKeyType::get(builder.getContext(), std::nullopt);
        Value rkGiant = builder.create<kmrt::LoadKeyOp>(loc, rotKeyType, giantAmt);

        // Apply giant step rotation
        Value ctGiant = builder.create<openfhe::RotOp>(
            loc, rotOp.getType(), cryptoContext, inputCt, rkGiant);

        // Clear giant step key
        builder.create<kmrt::ClearKeyOp>(loc, rkGiant);

        // Store for use in baby step loop
        giantStepMapping.map(rotOp.getResult(), ctGiant);
      }

      // Create baby step loop for remainder (0 to remainderSize)
      auto remainderLoop = builder.create<affine::AffineForOp>(
          loc, /*lbOperands=*/ValueRange{},
          /*lbMap=*/builder.getConstantAffineMap(0),
          /*ubOperands=*/ValueRange{},
          /*ubMap=*/builder.getConstantAffineMap(remainderSize),
          /*step=*/1, outerLoop.getResults(),
          [&](OpBuilder &remBuilder, Location remLoc, Value babyIV,
              ValueRange remIterArgs) {
            // Compute actual iteration index: giantOffset + babyIV
            auto indexMap = AffineMap::get(
                1, 0,
                remBuilder.getAffineDimExpr(0) + remBuilder.getAffineConstantExpr(giantOffset),
                remBuilder.getContext());
            Value actualIV = remBuilder.create<affine::AffineApplyOp>(
                remLoc, indexMap, ValueRange{babyIV});

            // Clone the loop body, replacing rotations with baby step only
            IRMapping mapping;
            mapping.map(loop.getInductionVar(), actualIV);
            for (auto [oldArg, newArg] :
                 llvm::zip(loop.getRegionIterArgs(), remIterArgs)) {
              mapping.map(oldArg, newArg);
            }

            // Collect operations to skip
            SmallVector<Operation *> opsToSkip;
            for (openfhe::RotOp rotOp : rotationsToDecompose) {
              if (auto loadOp = rotOp.getEvalKey().getDefiningOp<kmrt::LoadKeyOp>()) {
                opsToSkip.push_back(loadOp);
                if (auto indexCastOp = loadOp.getIndex().getDefiningOp<arith::IndexCastOp>()) {
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

              // Check if this is a rotation to decompose
              if (auto rotOp = dyn_cast<openfhe::RotOp>(op)) {
                if (llvm::find(rotationsToDecompose, rotOp) != rotationsToDecompose.end()) {
                  // Apply only baby step rotation
                  Value ctGiant = giantStepMapping.lookup(rotOp.getResult());
                  Value result = applyBabyStepRotation(remBuilder, rotOp, babyIV, ctGiant, mapping);
                  mapping.map(rotOp.getResult(), result);
                  continue;
                }
              }

              // Clone the operation normally
              Operation *clonedOp = remBuilder.clone(*op, mapping);
              for (auto [oldResult, newResult] :
                   llvm::zip(op->getResults(), clonedOp->getResults())) {
                mapping.map(oldResult, newResult);
              }
            }

            SmallVector<Value> mappedYields;
            for (Value yieldVal : yieldOperands) {
              mappedYields.push_back(mapping.lookup(yieldVal));
            }
            remBuilder.create<affine::AffineYieldOp>(remLoc, mappedYields);
          });

      finalLoop = remainderLoop;
    } else {
      finalLoop = outerLoop;
    }

    // Replace the original loop with the new nested loops
    loop.replaceAllUsesWith(finalLoop->getResults());
    loop.erase();
  }

  Value applyBabyStepRotation(OpBuilder &builder, openfhe::RotOp rotOp,
                              Value babyIV, Value ctGiant, IRMapping &mapping) {
    Location loc = rotOp.getLoc();
    Value cryptoContext = mapping.lookupOrDefault(rotOp.getCryptoContext());

    // === Baby Step Rotation Only ===
    // Load key for baby step - dynamic rotation (babyIV is runtime-determined)
    auto rotKeyType = kmrt::RotKeyType::get(builder.getContext(), std::nullopt);
    Value rkBaby = builder.create<kmrt::LoadKeyOp>(loc, rotKeyType, babyIV);

    // Use KMRT RotKey directly with OpenFHE rotation
    Value ctBaby = builder.create<openfhe::RotOp>(
        loc, rotOp.getType(), cryptoContext, ctGiant, rkBaby);

    // KMRT clear
    builder.create<kmrt::ClearKeyOp>(loc, rkBaby);

    return ctBaby;
  }
};

}  // namespace heir
}  // namespace mlir
