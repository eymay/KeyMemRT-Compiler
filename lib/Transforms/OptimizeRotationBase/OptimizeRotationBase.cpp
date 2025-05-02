#include "lib/Transforms/OptimizeRotationBase/OptimizeRotationBase.h"

#include "lib/Analysis/OptimizeRotationBaseAnalysis/OptimizeRotationBaseAnalysis.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/Support/LLVM.h"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_OPTIMIZEROTATION
#include "lib/Transforms/OptimizeRotationBase/OptimizeRotationBase.h.inc"

struct OptimizeRotation : impl::OptimizeRotationBase<OptimizeRotation> {
  using OptimizeRotationBase::OptimizeRotationBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    // Create and run the rotation base analysis
    RotationBaseAnalysis analysis(op, baseSetSize);
    if (failed(analysis.solve())) {
      return signalPassFailure();
    }

    // Output the results
    auto baseSet = analysis.getOptimalBaseSet();
    llvm::outs() << "Optimal rotation base set (size " << baseSet.size()
                 << "): ";
    for (auto idx : baseSet) {
      llvm::outs() << idx << " ";
    }
    llvm::outs() << "\n";

    // Output the composition of all non-base rotations
    llvm::outs() << "Compositions for non-base rotations:\n";
    int64_t totalOperations = 0;

    for (int64_t index : analysis.allRotationIndices) {
      if (analysis.isInBaseSet(index)) {
        llvm::outs() << index << " - Base rotation\n";
        continue;
      }

      auto composition = analysis.getComposition(index);
      llvm::outs() << index << " = ";

      bool firstTerm = true;
      for (const auto &entry : composition) {
        int64_t baseIdx = entry.first;
        int64_t count = entry.second;

        if (!firstTerm) {
          llvm::outs() << " + ";
        }

        if (count == 1) {
          llvm::outs() << baseIdx;
        } else {
          llvm::outs() << count << "*" << baseIdx;
        }

        firstTerm = false;
      }

      int64_t ops = analysis.getTotalOperations(index);
      totalOperations += ops;
      llvm::outs() << " (ops: " << ops << ")\n";
    }

    llvm::outs() << "Total operations for all non-base rotations: "
                 << totalOperations << "\n";

    // If requested, perform the actual transformation of the IR
    if (applyTransformation) {
      transformIR(analysis);
    }
  }

 private:
  void transformIR(const RotationBaseAnalysis &analysis) {
    // This would implement the actual IR transformation to replace
    // non-base rotation operations with sequences of base rotations
    //
    // For example, replace:
    //   %result = openfhe.rot %cc, %input, 4
    // With (for base set {1, 3}):
    //   %tmp1 = openfhe.rot %cc, %input, 1
    //   %tmp2 = openfhe.rot %cc, %tmp1, 3
    //   ... etc. following the composition

    llvm::outs() << "IR transformation is not implemented in this example.\n";

    // The full implementation would:
    // 1. Walk through all openfhe.rot operations
    // 2. For each operation with a non-base rotation index:
    //    a. Get the composition (sequence of base rotations)
    //    b. Create a sequence of rotations using only base indices
    //    c. Replace the original rotation with this sequence
  }
};

}  // namespace heir
}  // namespace mlir
