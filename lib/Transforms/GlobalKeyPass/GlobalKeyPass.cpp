#include "lib/Transforms/GlobalKeyPass/GlobalKeyPass.h"

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "llvm/include/llvm/ADT/DenseMap.h"             // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_GLOBALKEYPASS
#include "lib/Transforms/GlobalKeyPass/GlobalKeyPass.h.inc"

namespace {
using namespace openfhe;

// Template-based conversion pattern for deserialize_key ->
// deserialize_key_global
template <typename FromOp, typename ToOp>
struct ConvertDeserializeKeyOp : public OpRewritePattern<FromOp> {
  using OpRewritePattern<FromOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FromOp op,
                                PatternRewriter &rewriter) const override {
    // Extract rotation index from the index attribute
    auto indexAttr = op.getIndex();

    // Create the global version
    rewriter.create<ToOp>(op.getLoc(), op.getCryptoContext(), indexAttr);

    // Remove the original operation - we don't need its result anymore
    rewriter.eraseOp(op);

    return success();
  }
};

// Template-based conversion pattern for rot -> rotate_global
template <typename FromOp, typename ToOp>
struct ConvertRotateOp : public OpRewritePattern<FromOp> {
  using OpRewritePattern<FromOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FromOp op,
                                PatternRewriter &rewriter) const override {
    // Get the rotation index from the evaluation key
    auto evalKey = op.getEvalKey();
    auto evalKeyType = evalKey.getType();

    int64_t rotationIndex = evalKeyType.getRotationIndex();

    // Replace with global rotate - don't create deserialize, that's handled
    // separately
    rewriter.replaceOpWithNewOp<ToOp>(op, op.getOutput().getType(),
                                      op.getCryptoContext(), op.getCiphertext(),
                                      rewriter.getIndexAttr(rotationIndex));

    return success();
  }
};

// Template-based conversion pattern for clear_key -> clear_key_global
template <typename FromOp, typename ToOp>
struct ConvertClearKeyOp : public OpRewritePattern<FromOp> {
  using OpRewritePattern<FromOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FromOp op,
                                PatternRewriter &rewriter) const override {
    // Get the rotation index from the evaluation key
    auto evalKey = op.getEvalKey();
    auto evalKeyType = evalKey.getType();

    int64_t rotationIndex = evalKeyType.getRotationIndex();

    // Replace with global clear
    rewriter.replaceOpWithNewOp<ToOp>(op, op.getCryptoContext(),
                                      rewriter.getIndexAttr(rotationIndex));

    return success();
  }
};

struct GlobalKeyPass : impl::GlobalKeyPassBase<GlobalKeyPass> {
  using GlobalKeyPassBase::GlobalKeyPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Use templates to add conversion patterns
    patterns.add<ConvertRotateOp<RotOp, RotateGlobalOp>>(context);
    patterns.add<ConvertClearKeyOp<ClearKeyOp, ClearKeyGlobalOp>>(context);

    // Add pattern for standalone deserialize operations
    patterns
        .add<ConvertDeserializeKeyOp<DeserializeKeyOp, DeserializeKeyGlobalOp>>(
            context);

    // Apply patterns greedily
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace heir
}  // namespace mlir
