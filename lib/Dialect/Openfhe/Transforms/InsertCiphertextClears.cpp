#include "lib/Dialect/Openfhe/Transforms/InsertCiphertextClears.h"

#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "mlir/include/mlir/Analysis/Liveness.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

#define GEN_PASS_DEF_INSERTCIPHERTEXTCLEARS
#include "lib/Dialect/Openfhe/Transforms/Passes.h.inc"

namespace {

bool isCiphertextType(Type type) {
  return isa<lwe::NewLWECiphertextType, lwe::LWECiphertextType>(type);
}

struct InsertCiphertextClears
    : impl::InsertCiphertextClearsBase<InsertCiphertextClears> {
  using InsertCiphertextClearsBase::InsertCiphertextClearsBase;

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

    // Collect ciphertext values and find their last uses
    llvm::DenseMap<Value, Operation *> lastUseMap;

    // Walk all operations in the function
    funcOp.walk([&](Operation *op) {
      // Skip if this is already a clear operation
      if (isa<ClearCtOp>(op)) {
        return;
      }

      // Check each operand to see if this is its last use
      for (Value operand : op->getOperands()) {
        if (!isCiphertextType(operand.getType())) {
          continue;
        }

        // Don't clear function arguments
        if (isa<BlockArgument>(operand)) {
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
            lastUseMap[operand] = op;
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
            lastUseMap[operand] = op;
          }
        }
      }
    });

    // Insert clear operations after the last use of each ciphertext
    for (auto &entry : lastUseMap) {
      Value ciphertext = entry.first;
      Operation *lastUse = entry.second;

      builder.setInsertionPointAfter(lastUse);
      builder.create<ClearCtOp>(ciphertext.getLoc(), ciphertext);
    }
  }
};

}  // namespace

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
