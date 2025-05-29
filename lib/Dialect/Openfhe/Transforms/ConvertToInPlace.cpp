#include "lib/Dialect/Openfhe/Transforms/ConvertToInPlace.h"

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

#define GEN_PASS_DEF_CONVERTTOINPLACE
#include "lib/Dialect/Openfhe/Transforms/Passes.h.inc"

namespace {

// Check if it's safe to convert an operation to in-place form
bool isSafeToConvertInPlace(Operation *op) {
  // Check if the result is used by operations that would be compatible
  // with in-place modification of the first operand
  for (auto &use : op->getResult(0).getUses()) {
    Operation *user = use.getOwner();

    // If the user is a return operation, it's safe
    if (isa<func::ReturnOp>(user)) {
      continue;
    }

    // If the user is another OpenFHE operation that can work with modified
    // values
    if (isa<AddOp, SubOp, MulOp, AddPlainOp, SubPlainOp>(user)) {
      continue;
    }

    // For other operations, be conservative and don't convert
    return false;
  }

  return true;
}

// Only convert operations that we know for certain have in-place equivalents
// and that are already defined in the existing codebase

// Pattern for AddInPlaceOp (if it exists)
struct ConvertAddToInPlace : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter &rewriter) const override {
    if (!isSafeToConvertInPlace(op.getOperation())) {
      return failure();
    }

    // Create the in-place version using existing AddInPlaceOp
    rewriter.create<AddInPlaceOp>(op.getLoc(), op.getCryptoContext(),
                                  op.getLhs(), op.getRhs());

    rewriter.replaceAllUsesWith(op.getResult(), op.getLhs());
    rewriter.eraseOp(op);

    return success();
  }
};

// Pattern for SubInPlaceOp (if it exists)
struct ConvertSubToInPlace : public OpRewritePattern<SubOp> {
  using OpRewritePattern<SubOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubOp op,
                                PatternRewriter &rewriter) const override {
    if (!isSafeToConvertInPlace(op.getOperation())) {
      return failure();
    }

    // Create the in-place version using existing SubInPlaceOp
    rewriter.create<SubInPlaceOp>(op.getLoc(), op.getCryptoContext(),
                                  op.getLhs(), op.getRhs());

    rewriter.replaceAllUsesWith(op.getResult(), op.getLhs());
    rewriter.eraseOp(op);

    return success();
  }
};

}  // namespace

struct ConvertToInPlace : impl::ConvertToInPlaceBase<ConvertToInPlace> {
  using ConvertToInPlaceBase::ConvertToInPlaceBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // Add ONLY the patterns for operations that we know exist
    // Start with just Add and Sub - the most basic ones
    patterns.add<ConvertAddToInPlace, ConvertSubToInPlace>(context);

    // Apply the patterns greedily
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
