#include "lib/Transforms/FunctionExtractor/FunctionExtractor.h"

#include "llvm/include/llvm/Support/raw_ostream.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Support/LLVM.h"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_FUNCTIONEXTRACTOR
#include "lib/Transforms/FunctionExtractor/FunctionExtractor.h.inc"

struct FunctionExtractor : impl::FunctionExtractorBase<FunctionExtractor> {
  using impl::FunctionExtractorBase<FunctionExtractor>::FunctionExtractorBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();

    if (functionName.empty()) {
      moduleOp.emitError("No target function name specified for extraction");
      return signalPassFailure();
    }

    // Find the target function
    func::FuncOp targetFunc = nullptr;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      if (funcOp.getName() == functionName) {
        targetFunc = funcOp;
        break;
      }
    }

    if (!targetFunc) {
      moduleOp.emitError() << "Function '" << functionName << "' not found";
      return signalPassFailure();
    }

    // Instead of creating a new module, modify the existing one
    // Remove all functions except the target function
    SmallVector<func::FuncOp> functionsToErase;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      if (funcOp != targetFunc) {
        functionsToErase.push_back(funcOp);
      }
    }

    // Erase all other functions
    for (auto funcToErase : functionsToErase) {
      funcToErase.erase();
    }

    // The modified module (now containing only the target function)
    // will be automatically printed by MLIR pass infrastructure
  }
};

std::unique_ptr<Pass> createFunctionExtractorPass() {
  return std::make_unique<FunctionExtractor>();
}

}  // namespace heir
}  // namespace mlir
