#include "lib/Dialect/CKKS/Conversions/CKKSToLWE/CKKSToLWE.h"

#include <utility>

#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWEPatterns.h"
#include "lib/Utils/RewriteUtils/RewriteUtils.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"            // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/Transforms/WalkPatternRewriteDriver.h"  // from @llvm-project

namespace mlir::heir::ckks {

#define GEN_PASS_DEF_CKKSTOLWE
#include "lib/Dialect/CKKS/Conversions/CKKSToLWE/CKKSToLWE.h.inc"

static void preserveLinearTransformAttrs(Operation *from, Operation *to) {
  if (auto startAttr =
          from->getAttrOfType<IntegerAttr>("heir.linear_transform_start")) {
    to->setAttr("heir.linear_transform_start", startAttr);
  }
  if (auto endAttr =
          from->getAttrOfType<IntegerAttr>("heir.linear_transform_end")) {
    to->setAttr("heir.linear_transform_end", endAttr);
  }
}

// Replace your existing Convert template usage with this enhanced version:
template <typename FromOp, typename ToOp>
struct ConvertWithAttrs : public OpRewritePattern<FromOp> {
  using OpRewritePattern<FromOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FromOp op,
                                PatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<ToOp>(op, op.getResult().getType(),
                                                   op.getLhs(), op.getRhs());

    // Preserve linear transform attributes
    preserveLinearTransformAttrs(op.getOperation(), newOp.getOperation());

    return success();
  }
};

struct CKKSToLWE : public impl::CKKSToLWEBase<CKKSToLWE> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    RewritePatternSet patterns(context);
    patterns
        .add<ConvertWithAttrs<AddOp, lwe::RAddOp>, Convert<SubOp, lwe::RSubOp>,
             Convert<NegateOp, lwe::RNegateOp>, Convert<MulOp, lwe::RMulOp>,
             lwe::ConvertExtract<ExtractOp, MulPlainOp, RotateOp> >(context);
    walkAndApplyPatterns(module, std::move(patterns));
  }
};

}  // namespace mlir::heir::ckks
