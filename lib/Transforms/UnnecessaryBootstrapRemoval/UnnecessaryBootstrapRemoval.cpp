#include "lib/Transforms/UnnecessaryBootstrapRemoval/UnnecessaryBootstrapRemoval.h"

#include <vector>

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Support/LLVM.h"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_UNNECESSARYBOOTSTRAPREMOVAL
#include "lib/Transforms/UnnecessaryBootstrapRemoval/UnnecessaryBootstrapRemoval.h.inc"

namespace {

struct BootstrapAnalysisInfo {
  openfhe::BootstrapOp op;
  unsigned inputTowers = 0;
  unsigned outputTowers = 0;
  bool hasInputTowers = false;
  bool hasOutputTowers = false;
};

struct UnnecessaryBootstrapRemoval
    : impl::UnnecessaryBootstrapRemovalBase<UnnecessaryBootstrapRemoval> {
  using impl::UnnecessaryBootstrapRemovalBase<
      UnnecessaryBootstrapRemoval>::UnnecessaryBootstrapRemovalBase;

  void runOnOperation() override {
    Operation* op = getOperation();

    // Step 1: Collect all bootstrap operations and their tower information
    std::vector<BootstrapAnalysisInfo> bootstrapInfos;

    op->walk([&](openfhe::BootstrapOp bootstrapOp) {
      BootstrapAnalysisInfo info;
      info.op = bootstrapOp;

      // Get input tower count from the operand's defining operation
      Value inputCiphertext = bootstrapOp.getCiphertext();
      if (auto defOp = inputCiphertext.getDefiningOp()) {
        if (auto inputTowersAttr =
                defOp->getAttrOfType<IntegerAttr>("result_towers")) {
          info.inputTowers = inputTowersAttr.getInt();
          info.hasInputTowers = true;
        }
      }

      // Get output tower count from the bootstrap operation's annotation
      if (auto outputTowersAttr =
              bootstrapOp->getAttrOfType<IntegerAttr>("result_towers")) {
        info.outputTowers = outputTowersAttr.getInt();
        info.hasOutputTowers = true;
      }

      bootstrapInfos.push_back(info);
    });

    // Step 2: Identify unnecessary bootstraps
    std::vector<openfhe::BootstrapOp> unnecessaryBootstraps;
    unsigned totalBootstraps = bootstrapInfos.size();

    for (const auto& info : bootstrapInfos) {
      // Only consider bootstraps with complete tower information
      if (info.hasInputTowers && info.hasOutputTowers) {
        if (info.inputTowers == info.outputTowers) {
          llvm::errs() << "Found unnecessary bootstrap: input="
                       << info.inputTowers
                       << " towers, output=" << info.outputTowers
                       << " towers\n";
          unnecessaryBootstraps.push_back(info.op);
        } else {
          llvm::errs() << "Keeping necessary bootstrap: input="
                       << info.inputTowers
                       << " towers, output=" << info.outputTowers
                       << " towers\n";
        }
      } else {
        llvm::errs() << "Bootstrap lacks tower annotations - skipping (input="
                     << info.hasInputTowers
                     << ", output=" << info.hasOutputTowers << ")\n";
      }
    }

    // Step 3: Remove unnecessary bootstrap operations
    unsigned removedCount = 0;
    for (auto bootstrapOp : unnecessaryBootstraps) {
      // Replace the bootstrap result with its input
      Value input = bootstrapOp.getCiphertext();
      Value output = bootstrapOp.getResult();

      // Replace all uses of the bootstrap output with the input
      output.replaceAllUsesWith(input);
      bootstrapOp->erase();
      removedCount++;

      llvm::errs() << "Removed unnecessary bootstrap operation\n";
    }

    // Step 4: Report results
    llvm::errs() << "Bootstrap removal summary:\n";
    llvm::errs() << "  Total bootstraps found: " << totalBootstraps << "\n";
    llvm::errs() << "  Unnecessary bootstraps removed: " << removedCount
                 << "\n";
    llvm::errs() << "  Remaining bootstraps: "
                 << (totalBootstraps - removedCount) << "\n";
  }
};

}  // namespace
}  // namespace heir
}  // namespace mlir
