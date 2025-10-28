#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/Transforms/InsertClearOps.h"
#include "mlir/include/mlir/Analysis/Liveness.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"       // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

#define GEN_PASS_DEF_INSERTCLEAROPS
#include "lib/Dialect/Openfhe/Transforms/Passes.h.inc"

namespace {

bool isCiphertextType(Type type) {
  return isa<lwe::NewLWECiphertextType, lwe::LWECiphertextType>(type);
}

bool isPlaintextType(Type type) {
  return isa<lwe::NewLWEPlaintextType, lwe::LWEPlaintextType>(type);
}

// Check if a value is defined inside an affine or scf for loop
bool isDefinedInLoop(Value value) {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    // Block arguments can be loop induction variables, check parent op
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (isa<affine::AffineForOp, scf::ForOp>(parentOp)) {
      return true;
    }
  } else {
    // Check if the defining operation is inside a loop
    Operation *defOp = value.getDefiningOp();
    if (!defOp) return false;

    Operation *parentOp = defOp->getParentOp();
    while (parentOp) {
      if (isa<affine::AffineForOp, scf::ForOp>(parentOp)) {
        return true;
      }
      parentOp = parentOp->getParentOp();
    }
  }
  return false;
}

// Find the outermost loop that contains the given operation
// Returns nullptr if the operation is not inside any loop
Operation *findOutermostEnclosingLoop(Operation *op) {
  Operation *outermostLoop = nullptr;
  Operation *parentOp = op->getParentOp();

  while (parentOp) {
    if (isa<affine::AffineForOp, scf::ForOp>(parentOp)) {
      outermostLoop = parentOp;
    }
    parentOp = parentOp->getParentOp();
  }

  return outermostLoop;
}

struct InsertClearOps : impl::InsertClearOpsBase<InsertClearOps> {
  using InsertClearOpsBase::InsertClearOpsBase;

  void runOnOperation() override {
    Operation *moduleOrFuncOp = getOperation();

    // Handle both module and function operations
    if (auto moduleOp = dyn_cast<ModuleOp>(moduleOrFuncOp)) {
      // If running on module, process all functions
      moduleOp.walk([&](func::FuncOp funcOp) { processFunctionOp(funcOp); });
    } else if (auto funcOp = dyn_cast<func::FuncOp>(moduleOrFuncOp)) {
      // If running directly on function
      processFunctionOp(funcOp);
    }
  }

 private:
  void processFunctionOp(func::FuncOp funcOp) {
    // Run liveness analysis on this function
    Liveness liveness(funcOp);

    OpBuilder builder(funcOp.getContext());

    // Collect ciphertext and plaintext values and find their last uses
    llvm::DenseMap<Value, Operation *> lastUseCiphertextMap;
    llvm::DenseMap<Value, Operation *> lastUsePlaintextMap;

    // Walk all operations in the function
    funcOp.walk([&](Operation *op) {
      // Skip if this is already a clear operation
      if (isa<ClearCtOp, ClearPtOp>(op)) {
        return;
      }

      // Check each operand to see if this is its last use
      for (Value operand : op->getOperands()) {
        bool isCiphertext = isCiphertextType(operand.getType());
        bool isPlaintext = isPlaintextType(operand.getType());

        if (!isCiphertext && !isPlaintext) {
          continue;
        }

        // Don't clear function arguments
        if (isa<BlockArgument>(operand)) {
          continue;
        }

        // Don't clear values defined inside loops
        if (isDefinedInLoop(operand)) {
          continue;
        }

        // Don't clear if this value is returned from the function
        bool isReturned = false;
        for (auto &use : operand.getUses()) {
          if (isa<func::ReturnOp>(use.getOwner())) {
            isReturned = true;
            break;
          }
        }
        if (isReturned) {
          continue;
        }

        // Check if this operand is live after this operation
        Block *currentBlock = op->getBlock();
        const LivenessBlockInfo *info = liveness.getLiveness(currentBlock);

        if (info) {
          // If the value is not live out of this operation, this is its last
          // use
          if (!info->isLiveOut(operand)) {
            if (isCiphertext) {
              lastUseCiphertextMap[operand] = op;
            } else if (isPlaintext) {
              lastUsePlaintextMap[operand] = op;
            }
          }
        } else {
          // Fallback: if we can't get liveness info, use simple heuristic
          // Check if this is the last use by looking at remaining uses
          bool isLastUse = true;
          for (auto &use : operand.getUses()) {
            Operation *userOp = use.getOwner();
            // If there's a use after this operation in the same block, it's not
            // the last use
            if (userOp != op && userOp->getBlock() == op->getBlock()) {
              if (userOp->isBeforeInBlock(op)) {
                continue;  // This use is before current op
              } else {
                isLastUse = false;  // There's a use after current op
                break;
              }
            }
          }
          if (isLastUse) {
            if (isCiphertext) {
              lastUseCiphertextMap[operand] = op;
            } else if (isPlaintext) {
              lastUsePlaintextMap[operand] = op;
            }
          }
        }
      }
    });

    // Insert clear operations after the last use of each ciphertext
    for (auto &entry : lastUseCiphertextMap) {
      Value ciphertext = entry.first;
      Operation *lastUse = entry.second;

      // If the last use is inside a loop, we need to place the clear after
      // the outermost enclosing loop to ensure safety
      Operation *insertionPoint = lastUse;
      Operation *enclosingLoop = findOutermostEnclosingLoop(lastUse);
      if (enclosingLoop) {
        insertionPoint = enclosingLoop;
      }

      builder.setInsertionPointAfter(insertionPoint);
      builder.create<ClearCtOp>(ciphertext.getLoc(), ciphertext);
    }

    // Insert clear operations after the last use of each plaintext
    for (auto &entry : lastUsePlaintextMap) {
      Value plaintext = entry.first;
      Operation *lastUse = entry.second;

      // If the last use is inside a loop, we need to place the clear after
      // the outermost enclosing loop to ensure safety
      Operation *insertionPoint = lastUse;
      Operation *enclosingLoop = findOutermostEnclosingLoop(lastUse);
      if (enclosingLoop) {
        insertionPoint = enclosingLoop;
      }

      builder.setInsertionPointAfter(insertionPoint);
      builder.create<ClearPtOp>(plaintext.getLoc(), plaintext);
    }
  }
};

}  // namespace

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
