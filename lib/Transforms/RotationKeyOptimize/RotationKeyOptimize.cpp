#include "lib/Transforms/RotationKeyOptimize/RotationKeyOptimize.h"

#include "lib/Analysis/RotationKeyLivenessAnalysis/RotationKeyLivenessAnalysis.h"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ROTATIONKEYOPTIMIZE
#include "lib/Transforms/RotationKeyOptimize/RotationKeyOptimize.h.inc"

struct RotationKeyOptimize
    : impl::RotationKeyOptimizeBase<RotationKeyOptimize> {
  void runOnOperation() override {
    auto *op = getOperation();

    // Run the liveness analysis
    RotationKeyLivenessAnalysis analysis(op);

    // For now just dump the results
    analysis.dump();

    // Future optimization work will go here
  }
};
}  // namespace heir
}  // namespace mlir
