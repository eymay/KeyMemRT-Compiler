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

    // Create a new module with just the target function
    auto context = moduleOp.getContext();
    auto newModule = ModuleOp::create(moduleOp.getLoc());

    OpBuilder builder(context);
    builder.setInsertionPointToStart(newModule.getBody());

    // Clone the target function into the new module
    auto clonedFunc = cast<func::FuncOp>(builder.clone(*targetFunc));

    // Print the new module to stdout
    newModule.print(llvm::outs());

    // Clean up
    newModule.erase();
  }
};

std::unique_ptr<Pass> createFunctionExtractorPass() {
  return std::make_unique<FunctionExtractor>();
}

}  // namespace heir
}  // namespace mlir
