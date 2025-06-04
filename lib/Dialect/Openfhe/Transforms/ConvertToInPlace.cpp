#include "lib/Dialect/Openfhe/Transforms/ConvertToInPlace.h"

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "mlir/include/mlir/Analysis/Liveness.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Dominance.h"             // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace openfhe {

#define GEN_PASS_DEF_CONVERTTOINPLACE
#include "lib/Dialect/Openfhe/Transforms/Passes.h.inc"

namespace {

// Check if a value has only one use
bool hasOnlyOneUse(Value value) { return value.hasOneUse(); }

// Check if the single use is in the same block (to avoid control flow issues)
bool isSingleUseInSameBlock(Value value, Block* currentBlock) {
  if (!hasOnlyOneUse(value)) return false;

  Operation* user = *value.getUsers().begin();
  return user->getBlock() == currentBlock;
}

// Check if LHS value is used after the current operation
bool isLHSUsedAfterOp(Value lhs, Operation* currentOp,
                      DominanceInfo& dominanceInfo) {
  // Check if any use of LHS comes after the current operation
  for (Operation* user : lhs.getUsers()) {
    if (user == currentOp) continue;  // Skip the current operation

    // Use MLIR's dominance analysis to check if current operation dominates the
    // user If current op dominates user, then user comes after current op
    if (dominanceInfo.dominates(currentOp, user)) {
      return true;  // LHS is used after current operation
    }
  }

  return false;  // LHS is not used after current operation
}

// Check if it's safe to modify the LHS operand in-place
bool isSafeToModifyInPlace(Value lhs, Operation* currentOp,
                           DominanceInfo& dominanceInfo) {
  // If LHS has only one use (this operation), it's safe to modify
  if (hasOnlyOneUse(lhs)) {
    return true;
  }

  // If LHS has multiple uses, check if all uses are before this operation
  // or are this operation itself
  return !isLHSUsedAfterOp(lhs, currentOp, dominanceInfo);
}

// Check if the result value can be safely replaced by the LHS
bool canReplaceResultWithLHS(Operation* op, Value lhs) {
  Value result = op->getResult(0);

  // Check all uses of the result
  for (Operation* user : result.getUsers()) {
    // Check if user is in the same block
    if (user->getBlock() != op->getBlock()) {
      return false;
    }

    // For return operations, it's fine - we can return the modified LHS value
    // The key insight: if LHS is not used after this op, we can safely modify
    // it and return the modified value
  }

  return true;
}

// Enhanced safety check for in-place conversion with specific operand
bool isSafeToConvertInPlace(Operation* op, Value operandToModify,
                            DominanceInfo& dominanceInfo) {
  if (op->getNumResults() != 1) return false;

  Value result = op->getResult(0);

  // Check if we can safely modify the specified operand
  if (!isSafeToModifyInPlace(operandToModify, op, dominanceInfo)) {
    return false;
  }

  // Check if we can replace result uses with the operand to modify
  if (!canReplaceResultWithLHS(op, operandToModify)) {
    return false;
  }

  // Additional check: make sure the operation is in a function body
  if (!op->getParentOfType<func::FuncOp>()) {
    return false;
  }

  return true;
}

// Original safety check (for binary ops where LHS is always the target)
bool isSafeToConvertInPlace(Operation* op, DominanceInfo& dominanceInfo) {
  Value lhs = op->getOperand(1);  // First operand after crypto context
  return isSafeToConvertInPlace(op, lhs, dominanceInfo);
}

// Pattern for AddInPlaceOp
struct ConvertAddToInPlace : public OpRewritePattern<AddOp> {
  ConvertAddToInPlace(MLIRContext* context, DominanceInfo& dominanceInfo)
      : OpRewritePattern<AddOp>(context), dominanceInfo(dominanceInfo) {}

  LogicalResult matchAndRewrite(AddOp op,
                                PatternRewriter& rewriter) const override {
    if (!isSafeToConvertInPlace(op.getOperation(), dominanceInfo)) {
      return failure();
    }

    // Create the in-place version
    rewriter.create<AddInPlaceOp>(op.getLoc(), op.getCryptoContext(),
                                  op.getLhs(), op.getRhs());

    // Replace all uses of the result with the LHS operand
    rewriter.replaceAllUsesWith(op.getResult(), op.getLhs());

    // Remove the original operation
    rewriter.eraseOp(op);

    return success();
  }

 private:
  DominanceInfo& dominanceInfo;
};

// Pattern for SubInPlaceOp
struct ConvertSubToInPlace : public OpRewritePattern<SubOp> {
  ConvertSubToInPlace(MLIRContext* context, DominanceInfo& dominanceInfo)
      : OpRewritePattern<SubOp>(context), dominanceInfo(dominanceInfo) {}

  LogicalResult matchAndRewrite(SubOp op,
                                PatternRewriter& rewriter) const override {
    if (!isSafeToConvertInPlace(op.getOperation(), dominanceInfo)) {
      return failure();
    }

    // Create the in-place version
    rewriter.create<SubInPlaceOp>(op.getLoc(), op.getCryptoContext(),
                                  op.getLhs(), op.getRhs());

    // Replace all uses of the result with the LHS operand
    rewriter.replaceAllUsesWith(op.getResult(), op.getLhs());

    // Remove the original operation
    rewriter.eraseOp(op);

    return success();
  }

 private:
  DominanceInfo& dominanceInfo;
};

// Pattern for MulConstOp -> MulConstInPlaceOp
struct ConvertMulConstToInPlace : public OpRewritePattern<MulConstOp> {
  ConvertMulConstToInPlace(MLIRContext* context, DominanceInfo& dominanceInfo)
      : OpRewritePattern<MulConstOp>(context), dominanceInfo(dominanceInfo) {}

  LogicalResult matchAndRewrite(MulConstOp op,
                                PatternRewriter& rewriter) const override {
    if (!isSafeToConvertInPlace(op.getOperation(), dominanceInfo)) {
      return failure();
    }

    // Create the in-place version
    rewriter.create<MulConstInPlaceOp>(op.getLoc(), op.getCryptoContext(),
                                       op.getCiphertext(), op.getConstant());

    // Replace all uses of the result with the ciphertext operand
    rewriter.replaceAllUsesWith(op.getResult(), op.getCiphertext());

    // Remove the original operation
    rewriter.eraseOp(op);

    return success();
  }

 private:
  DominanceInfo& dominanceInfo;
};

// Pattern for NegateOp -> NegateInPlaceOp
struct ConvertNegateToInPlace : public OpRewritePattern<NegateOp> {
  ConvertNegateToInPlace(MLIRContext* context, DominanceInfo& dominanceInfo)
      : OpRewritePattern<NegateOp>(context), dominanceInfo(dominanceInfo) {}

  LogicalResult matchAndRewrite(NegateOp op,
                                PatternRewriter& rewriter) const override {
    if (!isSafeToConvertInPlace(op.getOperation(), dominanceInfo)) {
      return failure();
    }

    // Create the in-place version
    rewriter.create<NegateInPlaceOp>(op.getLoc(), op.getCryptoContext(),
                                     op.getCiphertext());

    // Replace all uses of the result with the ciphertext operand
    rewriter.replaceAllUsesWith(op.getResult(), op.getCiphertext());

    // Remove the original operation
    rewriter.eraseOp(op);

    return success();
  }

 private:
  DominanceInfo& dominanceInfo;
};

// Pattern for AddPlainOp -> AddPlainInPlaceOp
struct ConvertAddPlainToInPlace : public OpRewritePattern<AddPlainOp> {
  ConvertAddPlainToInPlace(MLIRContext* context, DominanceInfo& dominanceInfo)
      : OpRewritePattern<AddPlainOp>(context), dominanceInfo(dominanceInfo) {}

  LogicalResult matchAndRewrite(AddPlainOp op,
                                PatternRewriter& rewriter) const override {
    // For AddPlainOp, we need to determine which operand is the ciphertext
    // Looking at the definition, both lhs and rhs can be plaintext or
    // ciphertext We want to modify the ciphertext operand in-place
    Value ciphertext, plaintext;

    // Check LHS first
    if (isa<lwe::NewLWECiphertextType>(op.getLhs().getType()) &&
        isa<lwe::NewLWEPlaintextType>(op.getRhs().getType())) {
      ciphertext = op.getLhs();
      plaintext = op.getRhs();
    }
    // Check RHS
    else if (isa<lwe::NewLWEPlaintextType>(op.getLhs().getType()) &&
             isa<lwe::NewLWECiphertextType>(op.getRhs().getType())) {
      ciphertext = op.getRhs();
      plaintext = op.getLhs();
    } else {
      // Either both are same type or unsupported combination
      return failure();
    }

    // Check if we can safely modify the ciphertext in-place
    if (!isSafeToConvertInPlace(op.getOperation(), ciphertext, dominanceInfo)) {
      return failure();
    }

    // Create the in-place version
    rewriter.create<AddPlainInPlaceOp>(op.getLoc(), op.getCryptoContext(),
                                       ciphertext, plaintext);

    // Replace all uses of the result with the ciphertext operand
    rewriter.replaceAllUsesWith(op.getResult(), ciphertext);

    // Remove the original operation
    rewriter.eraseOp(op);

    return success();
  }

 private:
  DominanceInfo& dominanceInfo;
};

// Pattern for SubPlainOp -> SubPlainInPlaceOp
struct ConvertSubPlainToInPlace : public OpRewritePattern<SubPlainOp> {
  ConvertSubPlainToInPlace(MLIRContext* context, DominanceInfo& dominanceInfo)
      : OpRewritePattern<SubPlainOp>(context), dominanceInfo(dominanceInfo) {}

  LogicalResult matchAndRewrite(SubPlainOp op,
                                PatternRewriter& rewriter) const override {
    // For SubPlainOp, we need to determine which operand is the ciphertext
    Value ciphertext, plaintext;
    bool ciphertextIsLhs = false;

    // Check if LHS is ciphertext and RHS is plaintext
    if (isa<lwe::NewLWECiphertextType>(op.getLhs().getType()) &&
        isa<lwe::NewLWEPlaintextType>(op.getRhs().getType())) {
      ciphertext = op.getLhs();
      plaintext = op.getRhs();
      ciphertextIsLhs = true;
    }
    // Check if LHS is plaintext and RHS is ciphertext
    else if (isa<lwe::NewLWEPlaintextType>(op.getLhs().getType()) &&
             isa<lwe::NewLWECiphertextType>(op.getRhs().getType())) {
      ciphertext = op.getRhs();
      plaintext = op.getLhs();
      ciphertextIsLhs = false;
    } else {
      // Either both are same type or unsupported combination
      return failure();
    }

    // For subtraction, we can only do in-place if ciphertext is the LHS
    // (ciphertext - plaintext), not (plaintext - ciphertext)
    if (!ciphertextIsLhs) {
      return failure();
    }

    // Check if we can safely modify the ciphertext in-place
    if (!isSafeToConvertInPlace(op.getOperation(), ciphertext, dominanceInfo)) {
      return failure();
    }

    // Create the in-place version
    rewriter.create<SubPlainInPlaceOp>(op.getLoc(), op.getCryptoContext(),
                                       ciphertext, plaintext);

    // Replace all uses of the result with the ciphertext operand
    rewriter.replaceAllUsesWith(op.getResult(), ciphertext);

    // Remove the original operation
    rewriter.eraseOp(op);

    return success();
  }

 private:
  DominanceInfo& dominanceInfo;
};
struct ConvertRelinToInPlace : public OpRewritePattern<RelinOp> {
  ConvertRelinToInPlace(MLIRContext* context, DominanceInfo& dominanceInfo)
      : OpRewritePattern<RelinOp>(context), dominanceInfo(dominanceInfo) {}

  LogicalResult matchAndRewrite(RelinOp op,
                                PatternRewriter& rewriter) const override {
    if (!isSafeToConvertInPlace(op.getOperation(), dominanceInfo)) {
      return failure();
    }

    // Create the in-place version
    rewriter.create<RelinInPlaceOp>(op.getLoc(), op.getCryptoContext(),
                                    op.getCiphertext());

    // Replace all uses of the result with the ciphertext operand
    rewriter.replaceAllUsesWith(op.getResult(), op.getCiphertext());

    // Remove the original operation
    rewriter.eraseOp(op);

    return success();
  }

 private:
  DominanceInfo& dominanceInfo;
};

// Pattern for RotOp -> RotInPlaceOp (self-assignment style)
struct ConvertRotToInPlace : public OpRewritePattern<RotOp> {
  ConvertRotToInPlace(MLIRContext* context, DominanceInfo& dominanceInfo)
      : OpRewritePattern<RotOp>(context), dominanceInfo(dominanceInfo) {}

  LogicalResult matchAndRewrite(RotOp op,
                                PatternRewriter& rewriter) const override {
    Value ciphertext = op.getCiphertext();
    Value evalKey = op.getEvalKey();
    Value cryptoContext = op.getCryptoContext();

    // Check if we can safely modify the ciphertext in-place
    if (!isSafeToConvertInPlace(op.getOperation(), ciphertext, dominanceInfo)) {
      return failure();
    }

    // Store the result value before creating the new operation
    Value result = op.getResult();

    // Create the in-place version (self-assignment style)
    // Make sure to capture all operands before erasing the original op
    rewriter.create<RotInPlaceOp>(op.getLoc(), cryptoContext, ciphertext,
                                  evalKey);

    // Replace all uses of the result with the original ciphertext operand
    rewriter.replaceAllUsesWith(result, ciphertext);

    // Remove the original operation
    rewriter.eraseOp(op);

    return success();
  }

 private:
  DominanceInfo& dominanceInfo;
};

// Pattern for MulOp -> MulInPlaceOp (self-assignment style)
struct ConvertMulToInPlace : public OpRewritePattern<MulOp> {
  ConvertMulToInPlace(MLIRContext* context, DominanceInfo& dominanceInfo)
      : OpRewritePattern<MulOp>(context), dominanceInfo(dominanceInfo) {}

  LogicalResult matchAndRewrite(MulOp op,
                                PatternRewriter& rewriter) const override {
    Value lhs = op.getLhs();

    // Check if we can safely modify the LHS ciphertext in-place
    if (!isSafeToConvertInPlace(op.getOperation(), lhs, dominanceInfo)) {
      return failure();
    }

    // Create the in-place version (self-assignment style)
    rewriter.create<MulInPlaceOp>(op.getLoc(), op.getCryptoContext(), lhs,
                                  op.getRhs());

    // Replace all uses of the result with the LHS operand
    rewriter.replaceAllUsesWith(op.getResult(), lhs);

    // Remove the original operation
    rewriter.eraseOp(op);

    return success();
  }

 private:
  DominanceInfo& dominanceInfo;
};

// Pattern for MulPlainOp -> MulPlainInPlaceOp (self-assignment style)
struct ConvertMulPlainToInPlace : public OpRewritePattern<MulPlainOp> {
  ConvertMulPlainToInPlace(MLIRContext* context, DominanceInfo& dominanceInfo)
      : OpRewritePattern<MulPlainOp>(context), dominanceInfo(dominanceInfo) {}

  LogicalResult matchAndRewrite(MulPlainOp op,
                                PatternRewriter& rewriter) const override {
    Value ciphertext = op.getCiphertext();

    // Check if we can safely modify the ciphertext in-place
    if (!isSafeToConvertInPlace(op.getOperation(), ciphertext, dominanceInfo)) {
      return failure();
    }

    // Create the in-place version (self-assignment style)
    rewriter.create<MulPlainInPlaceOp>(op.getLoc(), op.getCryptoContext(),
                                       ciphertext, op.getPlaintext());

    // Replace all uses of the result with the ciphertext operand
    rewriter.replaceAllUsesWith(op.getResult(), ciphertext);

    // Remove the original operation
    rewriter.eraseOp(op);

    return success();
  }

 private:
  DominanceInfo& dominanceInfo;
};

struct ConvertToInPlace : impl::ConvertToInPlaceBase<ConvertToInPlace> {
  using ConvertToInPlaceBase::ConvertToInPlaceBase;

  void runOnOperation() override {
    MLIRContext* context = &getContext();

    // Get dominance analysis for the operation
    DominanceInfo dominanceInfo(getOperation());

    RewritePatternSet patterns(context);

    // Add patterns for all the in-place operations we know exist
    patterns.add<ConvertAddToInPlace>(context, dominanceInfo);
    patterns.add<ConvertSubToInPlace>(context, dominanceInfo);
    patterns.add<ConvertMulConstToInPlace>(context, dominanceInfo);
    patterns.add<ConvertNegateToInPlace>(context, dominanceInfo);
    patterns.add<ConvertRelinToInPlace>(context, dominanceInfo);
    patterns.add<ConvertRotToInPlace>(context, dominanceInfo);
    patterns.add<ConvertMulPlainToInPlace>(context, dominanceInfo);
    patterns.add<ConvertMulToInPlace>(context, dominanceInfo);

    // Apply the patterns greedily
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
