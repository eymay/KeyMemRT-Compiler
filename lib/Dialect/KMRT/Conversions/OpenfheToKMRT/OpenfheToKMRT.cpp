#include "lib/Dialect/KMRT/Conversions/OpenfheToKMRT/OpenfheToKMRT.h"

#include "lib/Dialect/KMRT/IR/KMRTDialect.h"
#include "lib/Dialect/KMRT/IR/KMRTOps.h"
#include "lib/Dialect/KMRT/IR/KMRTTypes.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"

// Include HEIR's OpenFHE dialect headers
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"

namespace mlir {
namespace keymemrt {
namespace kmrt {

#define GEN_PASS_DEF_OPENFHETOKMRT
#include "lib/Dialect/KMRT/Conversions/OpenfheToKMRT/OpenfheToKMRT.h.inc"

namespace {

/// Convert OpenFHE rotation operations to KMRT operations with explicit key
/// management
class ConvertOpenfheRotation
    : public OpRewritePattern<mlir::heir::openfhe::RotOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::heir::openfhe::RotOp rotOp,
                                PatternRewriter &rewriter) const override {
    Location loc = rotOp.getLoc();

    // Get the rotation index from the attribute
    auto indexAttr = rotOp.getIndex();
    int64_t rotIndex = indexAttr.getInt();

    // Create a constant value for the index
    auto indexValue = rewriter.create<mlir::arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(rotIndex));

    // Create the rotation key type
    auto rotKeyType = kmrt::RotKeyType::get(rewriter.getContext(), rotIndex);

    // Create load_key operation
    auto loadKeyOp = rewriter.create<kmrt::LoadKeyOp>(loc, rotKeyType,
                                                      indexValue.getResult());

    // Create the KMRT rotation operation
    auto rotationOp = rewriter.create<kmrt::RotationOp>(
        loc, rotOp.getResult().getType(), rotOp.getCiphertext(),
        loadKeyOp.getRotKey());

    // Create clear_key operation after the rotation
    rewriter.create<kmrt::ClearKeyOp>(loc, loadKeyOp.getRotKey());

    // Replace the original operation with the rotation result
    rewriter.replaceOp(rotOp, rotationOp.getResult());

    return success();
  }
};

} // namespace

struct OpenfheToKMRT : public impl::OpenfheToKMRTBase<OpenfheToKMRT> {
  using OpenfheToKMRTBase::OpenfheToKMRTBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto func = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<ConvertOpenfheRotation>(context);

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace kmrt
} // namespace keymemrt
} // namespace mlir
