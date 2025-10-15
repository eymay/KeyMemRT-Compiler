// File: lib/Transforms/SymbolicBSGSDecomposition/SymbolicBSGSDecomposition.cpp

#include "lib/Transforms/SymbolicBSGSDecomposition/SymbolicBSGSDecomposition.h"

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "llvm/include/llvm/Support/Debug.h"
#include "mlir/include/mlir/Dialect/Affine/Analysis/Utils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/LoopUtils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
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

    Value iv = loop.getInductionVar();

    // Find rotation operations that depend on the IV
    SmallVector<openfhe::RotOp> rotationsToDecompose;

    loop.walk([&](openfhe::RotOp rotOp) {
      // Check if this rotation uses a dynamic key
      auto deserOp =
          rotOp.getEvalKey().getDefiningOp<openfhe::DeserializeKeyDynamicOp>();
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

    // Decompose each rotation
    OpBuilder builder(loop.getContext());
    for (auto rotOp : rotationsToDecompose) {
      decomposeRotation(builder, loop, rotOp, rangeSize);
    }
  }

  void decomposeRotation(OpBuilder &builder, affine::AffineForOp loop,
                         openfhe::RotOp rotOp, int64_t rangeSize) {
    Location loc = rotOp.getLoc();
    Value iv = loop.getInductionVar();
    Value inputCt = rotOp.getCiphertext();
    Value cryptoContext = rotOp.getCryptoContext();

    // Create symbolic N2 parameter
    builder.setInsertionPoint(loop);
    int64_t estimatedN2 = static_cast<int64_t>(std::sqrt(rangeSize));
    Value N2 = builder.create<arith::ConstantIndexOp>(loc, estimatedN2);

    // Move builder inside the loop, before the rotation
    builder.setInsertionPoint(rotOp);

    // === Compute BSGS indices ===

    // Giant step index: i floordiv N2
    auto giantIdxMap = AffineMap::get(
        1, 1,
        builder.getAffineDimExpr(0).floorDiv(builder.getAffineSymbolExpr(0)),
        builder.getContext());
    Value giantIdx = builder.create<affine::AffineApplyOp>(loc, giantIdxMap,
                                                           ValueRange{iv, N2});

    // Baby step index: i mod N2
    auto babyIdxMap = AffineMap::get(
        1, 1, builder.getAffineDimExpr(0) % builder.getAffineSymbolExpr(0),
        builder.getContext());
    Value babyIdx = builder.create<affine::AffineApplyOp>(loc, babyIdxMap,
                                                          ValueRange{iv, N2});

    // Giant step amount: giant_idx * N2
    auto giantAmtMap = AffineMap::get(
        1, 1, builder.getAffineDimExpr(0) * builder.getAffineSymbolExpr(0),
        builder.getContext());
    Value giantAmt = builder.create<affine::AffineApplyOp>(
        loc, giantAmtMap, ValueRange{giantIdx, N2});

    // === Giant Step Rotation ===
    Value giantAmtI32 =
        builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), giantAmt);

    auto evalKeyType = openfhe::EvalKeyType::get(builder.getContext(),
                                                 builder.getIndexAttr(0));

    Value ekGiant = builder.create<openfhe::DeserializeKeyDynamicOp>(
        loc, evalKeyType, cryptoContext, giantAmtI32);

    Value ctGiant = builder.create<openfhe::RotOp>(
        loc, rotOp.getType(), cryptoContext, inputCt, ekGiant);

    builder.create<openfhe::ClearKeyOp>(loc, cryptoContext, ekGiant);

    // === Baby Step Rotation ===
    Value babyAmtI32 =
        builder.create<arith::IndexCastOp>(loc, builder.getI32Type(), babyIdx);

    Value ekBaby = builder.create<openfhe::DeserializeKeyDynamicOp>(
        loc, evalKeyType, cryptoContext, babyAmtI32);

    Value ctBaby = builder.create<openfhe::RotOp>(
        loc, rotOp.getType(), cryptoContext, ctGiant, ekBaby);

    builder.create<openfhe::ClearKeyOp>(loc, cryptoContext, ekBaby);

    // Replace the original rotation
    rotOp.getResult().replaceAllUsesWith(ctBaby);

    // Find and remove the original deserialize and clear operations
    auto originalDeser =
        rotOp.getEvalKey().getDefiningOp<openfhe::DeserializeKeyDynamicOp>();
    if (originalDeser) {
      // Find the corresponding clear operation
      for (Operation *user : originalDeser.getResult().getUsers()) {
        if (user == rotOp.getOperation()) continue;
        if (auto clearOp = dyn_cast<openfhe::ClearKeyOp>(user)) {
          clearOp.erase();
          break;
        }
      }

      Value deserInput = originalDeser.getIndex();
      if (auto indexCast = deserInput.getDefiningOp<arith::IndexCastOp>()) {
        // Check if this index_cast has no other users
        if (indexCast->hasOneUse()) {
          indexCast.erase();
        }
      }
      originalDeser.erase();
    }

    rotOp.erase();

    llvm::dbgs() << "Decomposed rotation with N2 = " << estimatedN2 << "\n";
  }
};

}  // namespace heir
}  // namespace mlir
