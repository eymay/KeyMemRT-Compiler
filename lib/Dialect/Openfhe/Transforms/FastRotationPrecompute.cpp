#include "lib/Dialect/Openfhe/Transforms/FastRotationPrecompute.h"

#include <algorithm>
#include <cstdint>

#include "lib/Dialect/KMRT/IR/KMRTTypes.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/ADT/DenseMap.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/DenseSet.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/Visitors.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

#define DEBUG_TYPE "fast-rotation-precompute"

namespace mlir {
namespace heir {
namespace openfhe {

#define GEN_PASS_DEF_FASTROTATIONPRECOMPUTE
#include "lib/Dialect/Openfhe/Transforms/Passes.h.inc"

// Helper to find the innermost affine loop containing an operation
static affine::AffineForOp getInnermostAffineLoop(Operation* op) {
  Operation* current = op->getParentOp();

  while (current && !isa<func::FuncOp>(current)) {
    if (auto affineLoop = dyn_cast<affine::AffineForOp>(current)) {
      return affineLoop;
    }
    current = current->getParentOp();
  }

  return nullptr;
}

// Helper to check if a value is loop-invariant for a given affine loop
static bool isLoopInvariant(Value val, affine::AffineForOp loop) {
  if (!val || !loop) return false;

  // Block arguments from function are loop-invariant
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    return !loop->isAncestor(blockArg.getOwner()->getParentOp());
  }

  // Check if the defining op is outside the loop
  auto* definingOp = val.getDefiningOp();
  if (!definingOp) return false;

  return !loop->isAncestor(definingOp);
}

void processFunc(func::FuncOp funcOp, Value cryptoContext) {
  IRRewriter builder(funcOp->getContext());

  // Track ciphertexts we've already precomputed to avoid duplicates
  llvm::DenseMap<Value, Value> ciphertextToPrecompute;

  // Process rotations in a single walk
  funcOp->walk([&](RotOp rotOp) {
    Value ciphertext = rotOp.getCiphertext();

    // Check if we already have a precompute for this ciphertext
    if (ciphertextToPrecompute.count(ciphertext)) {
      // Replace this rotation with a fast rotation
      builder.setInsertionPoint(rotOp);
      auto fastRot = builder.create<FastRotationOp>(
          rotOp->getLoc(), rotOp.getType(), rotOp.getCryptoContext(),
          rotOp.getCiphertext(), rotOp.getEvalKey(),
          ciphertextToPrecompute[ciphertext]);
      builder.replaceOp(rotOp, fastRot.getResult());
      return;
    }

    // Check if this rotation is inside an affine loop
    auto innermostLoop = getInnermostAffineLoop(rotOp);

    bool shouldOptimize = false;
    Operation* insertionPoint = rotOp;

    if (innermostLoop && isLoopInvariant(ciphertext, innermostLoop)) {
      // Ciphertext is loop-invariant with respect to the innermost loop.
      // Find the outermost loop where it's still invariant for maximum hoisting.
      affine::AffineForOp outermostInvariantLoop = innermostLoop;
      Operation* current = innermostLoop->getParentOp();

      while (current && !isa<func::FuncOp>(current)) {
        if (auto affineLoop = dyn_cast<affine::AffineForOp>(current)) {
          if (isLoopInvariant(ciphertext, affineLoop)) {
            outermostInvariantLoop = affineLoop;
          } else {
            break; // Stop if we find a loop where it's not invariant
          }
        }
        current = current->getParentOp();
      }

      shouldOptimize = true;
      insertionPoint = outermostInvariantLoop;
      LLVM_DEBUG(llvm::dbgs()
                 << "Found loop-invariant rotation, hoisting precompute before "
                 << (outermostInvariantLoop == innermostLoop ? "innermost" : "outermost invariant")
                 << " loop\n");
    } else {
      // Check if there are multiple rotations using the same ciphertext
      int rotationCount = 0;
      funcOp->walk([&](RotOp otherRot) {
        if (otherRot.getCiphertext() == ciphertext) {
          rotationCount++;
        }
      });

      if (rotationCount >= 2) {
        shouldOptimize = true;
        LLVM_DEBUG(llvm::dbgs() << "Found " << rotationCount
                                << " rotations on same ciphertext, creating precompute\n");
      }
    }

    if (shouldOptimize) {
      // Create the digit decomposition type
      auto digitDecompType = openfhe::DigitDecompositionType::get(builder.getContext());

      // Insert precompute at the appropriate location
      builder.setInsertionPoint(insertionPoint);
      auto precomputeOp = builder.create<FastRotationPrecomputeOp>(
          ciphertext.getLoc(), digitDecompType, cryptoContext, ciphertext);

      // Store for future rotations on this ciphertext
      ciphertextToPrecompute[ciphertext] = precomputeOp.getResult();

      // Replace this rotation with fast rotation
      builder.setInsertionPoint(rotOp);
      auto fastRot = builder.create<FastRotationOp>(
          rotOp->getLoc(), rotOp.getType(), rotOp.getCryptoContext(),
          rotOp.getCiphertext(), rotOp.getEvalKey(), precomputeOp.getResult());
      builder.replaceOp(rotOp, fastRot.getResult());
    }
  });
}

struct FastRotationPrecompute
    : impl::FastRotationPrecomputeBase<FastRotationPrecompute> {
  using FastRotationPrecomputeBase::FastRotationPrecomputeBase;

  void runOnOperation() override {
    // Process each function separately to avoid cross-function batching
    getOperation()->walk([&](func::FuncOp op) -> WalkResult {
      auto result = getContextualArgFromFunc<openfhe::CryptoContextType>(op);
      if (failed(result)) {
        LLVM_DEBUG(llvm::dbgs() << "Skipping func with no cryptocontext arg: "
                                << op.getSymName() << "\n");
        return WalkResult::advance();
      }
      Value cryptoContext = result.value();
      processFunc(op, cryptoContext);
      return WalkResult::advance();
    });
  }
};

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
