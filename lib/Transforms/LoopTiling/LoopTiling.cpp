#include "lib/Transforms/LoopTiling/LoopTiling.h"

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "llvm/include/llvm/Support/Debug.h"
#include "mlir/include/mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/include/mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/include/mlir/Dialect/Affine/Utils.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/IRMapping.h"
#include "mlir/include/mlir/IR/IntegerSet.h"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_LOOPTILING
#include "lib/Transforms/LoopTiling/LoopTiling.h.inc"

// Helper to find the N2 constant used in BSGS decomposition
static std::optional<int64_t> findBSGSTileSize(affine::AffineForOp loop) {
  // Look for the constant that appears in the affine.apply operations
  // The BSGS decomposition uses: i floordiv N2, i mod N2, etc.

  std::optional<int64_t> tileSize;

  loop.walk([&](affine::AffineApplyOp applyOp) {
    // Check if this is a floordiv or mod operation with a symbol
    AffineMap map = applyOp.getAffineMap();
    if (map.getNumSymbols() == 1 && applyOp.getMapOperands().size() == 2) {
      // The second operand should be the N2 constant
      Value symbolOperand = applyOp.getMapOperands()[1];
      if (auto constOp =
              symbolOperand.getDefiningOp<arith::ConstantIndexOp>()) {
        tileSize = constOp.value();
      }
    }
  });

  return tileSize;
}

struct LoopTiling : impl::LoopTilingBase<LoopTiling> {
  using LoopTilingBase::LoopTilingBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    llvm::dbgs() << "=== Loop Tiling for BSGS ===\n";

    // Collect loops that need tiling
    SmallVector<affine::AffineForOp> loopsToTile;
    func.walk([&](affine::AffineForOp loop) {
      // Only tile loops that have BSGS pattern (contain rotations)
      bool hasRotations = false;
      loop.walk([&](openfhe::RotOp) { hasRotations = true; });

      if (hasRotations && loop.hasConstantBounds()) {
        auto tileSize = findBSGSTileSize(loop);
        if (tileSize) {
          loopsToTile.push_back(loop);
        }
      }
    });

    for (auto loop : loopsToTile) {
      // Skip if loop was already erased
      if (!loop->getBlock()) continue;

      auto tileSize = findBSGSTileSize(loop);
      if (!tileSize) {
        llvm::dbgs() << "Could not determine tile size, skipping\n";
        continue;
      }

      llvm::dbgs() << "Tiling loop with tile size: " << *tileSize << "\n";

      if (succeeded(tileLoop(loop, *tileSize))) {
        llvm::dbgs() << "Successfully tiled loop\n";
      } else {
        llvm::dbgs() << "Failed to tile loop\n";
      }
    }

    llvm::dbgs() << "Loop tiling completed\n";
  }

 private:
  LogicalResult tileLoop(affine::AffineForOp loop, int64_t tileSize) {
    OpBuilder builder(loop.getContext());
    Location loc = loop.getLoc();

    int64_t lb = loop.getConstantLowerBound();
    int64_t ub = loop.getConstantUpperBound();
    int64_t step = loop.getStepAsInt();

    // Calculate the split point: how many complete tiles fit
    int64_t rangeSize = ub - lb;
    int64_t numCompleteTiles = rangeSize / tileSize;
    int64_t tiledUpperBound = lb + numCompleteTiles * tileSize;
    int64_t remainder = rangeSize % tileSize;

    llvm::dbgs() << "  Loop range: " << lb << " to " << ub << " (step " << step
                 << ")\n";
    llvm::dbgs() << "  Complete tiles: " << numCompleteTiles
                 << ", tiled range: " << lb << " to " << tiledUpperBound
                 << ", remainder: " << remainder << "\n";

    builder.setInsertionPoint(loop);

    // Get iter args from original loop
    SmallVector<Value> iterOperands(loop.getInits().begin(),
                                    loop.getInits().end());

    Value lastResult;

    // Create the main tiled loops (for complete tiles)
    if (numCompleteTiles > 0) {
      auto outerLoop = builder.create<affine::AffineForOp>(
          loc, 0, numCompleteTiles, 1, iterOperands,
          [&](OpBuilder &outerBuilder, Location outerLoc, Value outerIV,
              ValueRange outerIterArgs) {
            // Compute the base index for this tile: i_base = lb + outerIV * tileSize
            AffineExpr d0 = outerBuilder.getAffineDimExpr(0);  // outerIV
            AffineExpr s0 = outerBuilder.getAffineSymbolExpr(0);  // lb
            AffineExpr s1 = outerBuilder.getAffineSymbolExpr(1);  // tileSize
            AffineMap baseIndexMap = AffineMap::get(1, 2, s0 + d0 * s1, builder.getContext());

            Value lbValue = outerBuilder.create<arith::ConstantIndexOp>(outerLoc, lb);
            Value tileSizeValue = outerBuilder.create<arith::ConstantIndexOp>(outerLoc, tileSize);

            Value baseIndex = outerBuilder.create<affine::AffineApplyOp>(
                outerLoc, baseIndexMap, ValueRange{outerIV, lbValue, tileSizeValue});

            // Create the inner loop first (without hoisting)
            auto innerLoop = outerBuilder.create<affine::AffineForOp>(
                outerLoc, 0, tileSize, step, outerIterArgs,
                [&](OpBuilder &innerBuilder, Location innerLoc, Value innerIV,
                    ValueRange innerIterArgs) {
                  // Compute the actual loop index: i = baseIndex + innerIV
                  AffineExpr d0 = innerBuilder.getAffineDimExpr(0);  // baseIndex
                  AffineExpr d1 = innerBuilder.getAffineDimExpr(1);  // innerIV
                  AffineMap actualIndexMap = AffineMap::get(2, 0, d0 + d1, builder.getContext());

                  Value actualIndex = innerBuilder.create<affine::AffineApplyOp>(
                      innerLoc, actualIndexMap, ValueRange{baseIndex, innerIV});

                  // Clone the original loop body
                  IRMapping mapping;
                  mapping.map(loop.getInductionVar(), actualIndex);

                  // Map iter args
                  for (auto [oldArg, newArg] :
                       llvm::zip(loop.getRegionIterArgs(), innerIterArgs)) {
                    mapping.map(oldArg, newArg);
                  }

                  // Clone all operations
                  for (Operation &op : loop.getBody()->without_terminator()) {
                    innerBuilder.clone(op, mapping);
                  }

                  // Get the yield value
                  auto origYield =
                      cast<affine::AffineYieldOp>(loop.getBody()->getTerminator());

                  SmallVector<Value> yieldValues;
                  for (Value operand : origYield.getOperands()) {
                    yieldValues.push_back(mapping.lookup(operand));
                  }
                  innerBuilder.create<affine::AffineYieldOp>(innerLoc, yieldValues);
                });

            // Now analyze the inner loop body to find operations that don't use innerIV
            Block *innerBody = innerLoop.getBody();
            Value innerIV = innerLoop.getInductionVar();

            SmallVector<Operation *> toHoist;
            DenseSet<Value> innerIVDependents;  // Values that depend on innerIV

            // Mark innerIV itself
            innerIVDependents.insert(innerIV);
            llvm::dbgs() << "  Inner loop IV: " << innerIV << "\n";

            // Propagate dependencies
            bool changed = true;
            while (changed) {
              changed = false;
              for (Operation &op : innerBody->without_terminator()) {
                // Check if any operand depends on innerIV
                bool usesInnerIV = false;
                for (Value operand : op.getOperands()) {
                  if (innerIVDependents.contains(operand)) {
                    usesInnerIV = true;
                    break;
                  }
                }

                if (usesInnerIV) {
                  // Mark all results as dependent
                  for (Value result : op.getResults()) {
                    if (!innerIVDependents.contains(result)) {
                      innerIVDependents.insert(result);
                      changed = true;
                    }
                  }
                }
              }
            }

            // Identify operations to hoist: those whose results are NOT innerIV-dependent
            // AND all their users are also not innerIV-dependent (to maintain dominance)
            DenseSet<Operation *> canBeHoisted;

            for (Operation &op : innerBody->without_terminator()) {
              if (isa<affine::AffineYieldOp>(&op)) continue;

              bool canHoist = true;

              // Check if any result is innerIV-dependent
              for (Value result : op.getResults()) {
                if (innerIVDependents.contains(result)) {
                  canHoist = false;
                  break;
                }
              }

              // Don't hoist if it uses inner loop iter args
              for (Value operand : op.getOperands()) {
                for (auto iterArg : innerLoop.getRegionIterArgs()) {
                  if (operand == iterArg) {
                    canHoist = false;
                    break;
                  }
                }
              }

              if (canHoist) {
                canBeHoisted.insert(&op);
                llvm::dbgs() << "    Can hoist: " << op.getName() << "\n";
              } else {
                llvm::dbgs() << "    Cannot hoist: " << op.getName() << "\n";
              }
            }

            // Hoist ALL operations in canBeHoisted, in program order to maintain dependencies
            for (Operation &op : innerBody->without_terminator()) {
              if (canBeHoisted.contains(&op)) {
                toHoist.push_back(&op);
              }
            }

            llvm::dbgs() << "  Found " << toHoist.size() << " operations to hoist from inner to outer loop\n";

            if (!toHoist.empty()) {
              // Move operations from inner loop to outer loop
              IRMapping hoistMapping;

              for (Operation *op : toHoist) {
                // Clone to outer loop
                Operation *hoisted = outerBuilder.clone(*op);

                // Map results
                for (auto [origResult, hoistedResult] :
                     llvm::zip(op->getResults(), hoisted->getResults())) {
                  hoistMapping.map(origResult, hoistedResult);
                }
              }

              // Update uses in the inner loop
              for (Operation &op : innerBody->without_terminator()) {
                for (unsigned i = 0; i < op.getNumOperands(); ++i) {
                  Value operand = op.getOperand(i);
                  if (Value mapped = hoistMapping.lookupOrNull(operand)) {
                    op.setOperand(i, mapped);
                  }
                }
              }

              // Remove hoisted operations from inner loop
              for (Operation *op : llvm::reverse(toHoist)) {
                op->erase();
              }
            }

            // Yield from outer loop
            outerBuilder.create<affine::AffineYieldOp>(outerLoc,
                                                       innerLoop.getResults());
          });

      lastResult = outerLoop.getResult(0);
    } else {
      // No complete tiles, use the initial value
      lastResult = iterOperands[0];
    }

    // Create the remainder loop (peeled iterations)
    if (remainder > 0) {
      // For the remainder loop, we need to hoist the giant step computation
      // because it's loop-invariant (all remainder iterations have the same giant step).

      // The giant step is constant for all remainder iterations:
      // giant_step = tiledUpperBound floordiv tileSize
      int64_t giantStepIndex = tiledUpperBound / tileSize;
      int64_t giantRotationAmount = giantStepIndex * tileSize;

      llvm::dbgs() << "  Remainder loop: hoisting giant step " << giantStepIndex
                   << " (rotation " << giantRotationAmount << ")\n";

      // Find the crypto_context and input ciphertext from the original loop
      Value cryptoContext;
      Value inputCiphertext;
      Type evalKeyType;
      Type ciphertextType;

      loop.walk([&](openfhe::RotOp rotOp) {
        if (!cryptoContext) {
          cryptoContext = rotOp.getCryptoContext();
          inputCiphertext = rotOp.getCiphertext();
          ciphertextType = rotOp.getType();
        }
      });

      loop.walk([&](openfhe::DeserializeKeyDynamicOp deserOp) {
        if (!evalKeyType) {
          evalKeyType = deserOp.getType();
        }
      });

      // Hoist the giant step rotation computation before the remainder loop
      builder.setInsertionPoint(builder.getInsertionBlock(),
                                builder.getInsertionPoint());

      Value giantRotIdx = builder.create<arith::ConstantIndexOp>(
          loc, giantRotationAmount);
      Value giantRotIdxI32 =
          builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), giantRotIdx);

      // Deserialize the giant step key and perform rotation
      Value hoistedEvalKey = builder.create<openfhe::DeserializeKeyDynamicOp>(
          loc, evalKeyType, cryptoContext, giantRotIdxI32);

      Value hoistedCiphertext = builder.create<openfhe::RotOp>(
          loc, ciphertextType, cryptoContext, inputCiphertext, hoistedEvalKey);

      builder.create<openfhe::ClearKeyOp>(loc, cryptoContext, hoistedEvalKey);

      // Now create the remainder loop with the hoisted giant step
      auto remainderLoop = builder.create<affine::AffineForOp>(
          loc, tiledUpperBound, ub, step, ValueRange{lastResult},
          [&](OpBuilder &remBuilder, Location remLoc, Value remIV,
              ValueRange remIterArgs) {
            // Clone the original loop body, skipping giant step operations
            IRMapping mapping;
            mapping.map(loop.getInductionVar(), remIV);

            // Map iter args
            for (auto [oldArg, newArg] :
                 llvm::zip(loop.getRegionIterArgs(), remIterArgs)) {
              mapping.map(oldArg, newArg);
            }

            // We need to identify and skip the giant step operations:
            // 1. The first affine.apply computing floordiv
            // 2. The affine.apply computing the multiplication
            // 3. The index_cast for the giant step
            // 4. The first deserialize_key_dynamic
            // 5. The first rot operation
            // 6. The first clear_key operation

            int rotCount = 0;
            int deserializeCount = 0;
            int clearCount = 0;

            for (Operation &op : loop.getBody()->without_terminator()) {
              bool shouldSkip = false;

              // Skip the first rotation (giant step)
              if (auto rotOp = dyn_cast<openfhe::RotOp>(&op)) {
                if (rotCount == 0) {
                  // Map the result to our hoisted ciphertext
                  mapping.map(rotOp.getResult(), hoistedCiphertext);
                  shouldSkip = true;
                }
                rotCount++;
              }

              // Skip the first deserialize (for giant step key)
              if (auto deserOp = dyn_cast<openfhe::DeserializeKeyDynamicOp>(&op)) {
                if (deserializeCount == 0) {
                  shouldSkip = true;
                }
                deserializeCount++;
              }

              // Skip the first clear_key (for giant step key)
              if (auto clearOp = dyn_cast<openfhe::ClearKeyOp>(&op)) {
                if (clearCount == 0) {
                  shouldSkip = true;
                }
                clearCount++;
              }

              // Skip affine.apply and index_cast operations related to giant step
              if (auto applyOp = dyn_cast<affine::AffineApplyOp>(&op)) {
                // Check if any users lead to the first deserialize
                for (auto *user : applyOp.getResult().getUsers()) {
                  if (auto castOp = dyn_cast<arith::IndexCastOp>(user)) {
                    for (auto *castUser : castOp.getResult().getUsers()) {
                      if (auto deserOp = dyn_cast<openfhe::DeserializeKeyDynamicOp>(castUser)) {
                        // Check if this is the first deserialize
                        bool isFirst = true;
                        for (Operation &checkOp : loop.getBody()->without_terminator()) {
                          if (&checkOp == deserOp) break;
                          if (isa<openfhe::DeserializeKeyDynamicOp>(&checkOp)) {
                            isFirst = false;
                            break;
                          }
                        }
                        if (isFirst) shouldSkip = true;
                      }
                    }
                  }
                }
              }

              if (auto castOp = dyn_cast<arith::IndexCastOp>(&op)) {
                // Check if this cast feeds into the first deserialize
                for (auto *user : castOp.getResult().getUsers()) {
                  if (auto deserOp = dyn_cast<openfhe::DeserializeKeyDynamicOp>(user)) {
                    bool isFirst = true;
                    for (Operation &checkOp : loop.getBody()->without_terminator()) {
                      if (&checkOp == deserOp) break;
                      if (isa<openfhe::DeserializeKeyDynamicOp>(&checkOp)) {
                        isFirst = false;
                        break;
                      }
                    }
                    if (isFirst) shouldSkip = true;
                  }
                }
              }

              if (!shouldSkip) {
                remBuilder.clone(op, mapping);
              }
            }

            // Get the yield value from the cloned operations
            auto origYield =
                cast<affine::AffineYieldOp>(loop.getBody()->getTerminator());

            SmallVector<Value> yieldValues;
            for (Value operand : origYield.getOperands()) {
              yieldValues.push_back(mapping.lookup(operand));
            }
            remBuilder.create<affine::AffineYieldOp>(remLoc, yieldValues);
          });

      lastResult = remainderLoop.getResult(0);
    }

    // Replace the original loop with the final result
    loop.replaceAllUsesWith(ValueRange{lastResult});
    loop.erase();

    return success();
  }
};

}  // namespace heir
}  // namespace mlir
