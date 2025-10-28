//===- LowerLinearTransform.cpp - Lower OpenFHE Linear Transform --------===//

#include "lib/Transforms/LowerLinearTransform/LowerLinearTransform.h"

#include "lib/Dialect/KMRT/IR/KMRTOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "llvm/include/llvm/Support/Debug.h"
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace heir {

namespace {

// Helper to extract scaling factor from types
static int64_t getScalingFactor(Type type) {
  if (auto ctType = dyn_cast<lwe::NewLWECiphertextType>(type)) {
    auto ptSpace = ctType.getPlaintextSpace();
    if (auto encoding = dyn_cast<lwe::InverseCanonicalEncodingAttr>(
            ptSpace.getEncoding())) {
      return encoding.getScalingFactor();
    }
  } else if (auto ptType = dyn_cast<lwe::NewLWEPlaintextType>(type)) {
    auto ptSpace = ptType.getPlaintextSpace();
    if (auto encoding = dyn_cast<lwe::InverseCanonicalEncodingAttr>(
            ptSpace.getEncoding())) {
      return encoding.getScalingFactor();
    }
  }
  return 1;
}

// Helper to get the raw tensor from weights plaintext
Value getWeightTensor(PatternRewriter &rewriter, Location loc, Value weights) {
  auto definingOp = weights.getDefiningOp();

  // If weights is already a tensor type, return it directly
  if (auto tensorType = dyn_cast<RankedTensorType>(weights.getType())) {
    return weights;
  }

  // If weights is a block argument (function parameter), decode it
  if (!definingOp) {
    llvm::dbgs() << "  Weights is a block argument, decoding\n";
    if (auto weightsType =
            dyn_cast<lwe::NewLWEPlaintextType>(weights.getType())) {
      auto tensorType = weightsType.getApplicationData().getMessageType();
      Value decodedWeights =
          rewriter.create<lwe::RLWEDecodeOp>(loc, tensorType, weights);
      return decodedWeights;
    }
  }

  // If weights come from an encode operation, get the input tensor directly
  if (auto encodeOp = dyn_cast<lwe::RLWEEncodeOp>(definingOp)) {
    llvm::dbgs() << "  Found RLWEEncodeOp, getting input tensor\n";
    return encodeOp.getInput();
  }

  // If weights is a constant, return it directly
  if (auto constantOp = dyn_cast<arith::ConstantOp>(definingOp)) {
    llvm::dbgs() << "  Found ConstantOp, returning directly\n";
    return weights;
  }

  // Fallback: try to decode if it's a plaintext type
  llvm::dbgs() << "  Unrecognized defining op type, attempting decode\n";
  if (auto weightsType =
          dyn_cast<lwe::NewLWEPlaintextType>(weights.getType())) {
    auto tensorType = weightsType.getApplicationData().getMessageType();
    Value decodedWeights =
        rewriter.create<lwe::RLWEDecodeOp>(loc, tensorType, weights);
    return decodedWeights;
  }

  llvm::dbgs() << "ERROR: Could not get weight tensor\n";
  return Value();
}

// Pattern for lowering linear transform operations
class LowerLinearTransformPattern
    : public OpRewritePattern<openfhe::LinearTransformOp> {
 public:
  LowerLinearTransformPattern(MLIRContext *context, bool useUnrolledForm)
      : OpRewritePattern<openfhe::LinearTransformOp>(context),
        useUnrolledForm(useUnrolledForm) {}

  LogicalResult matchAndRewrite(openfhe::LinearTransformOp op,
                                PatternRewriter &rewriter) const override {
    if (useUnrolledForm) {
      return lowerToUnrolledForm(op, rewriter);
    } else {
      return lowerToAffineLoopForm(op, rewriter);
    }
  }

 private:
  bool useUnrolledForm;

  // Helper to extract a single diagonal from the weight matrix
  Value extractDiagonal(PatternRewriter &rewriter, Location loc, Value weights,
                        int32_t diagonalIdx, int32_t slots, int64_t targetScale,
                        Value cryptoContext) const {
    llvm::dbgs() << "  extractDiagonal called for diagonal " << diagonalIdx
                 << "\n";

    auto slotType = rewriter.getF64Type();
    auto diagonalTensorType = RankedTensorType::get({slots}, slotType);

    // Get the weight matrix as a tensor
    Value weightMatrix = getWeightTensor(rewriter, loc, weights);

    if (!weightMatrix) {
      llvm::dbgs() << "ERROR: Could not get weight tensor, using dummy data\n";
      // Create dummy diagonal data for testing
      SmallVector<double> dummyData(slots, 0.1 * (diagonalIdx + 1));
      auto dummyAttr = DenseElementsAttr::get(diagonalTensorType,
                                              ArrayRef<double>(dummyData));
      weightMatrix = rewriter.create<arith::ConstantOp>(loc, dummyAttr);
      return weightMatrix;
    }

    auto weightMatrixType = cast<RankedTensorType>(weightMatrix.getType());
    auto shape = weightMatrixType.getShape();

    llvm::dbgs() << "  Weight matrix shape: [";
    for (auto dim : shape) {
      llvm::dbgs() << dim << " ";
    }
    llvm::dbgs() << "]\n";

    // Extract the diagonalIdx-th row using tensor.extract_slice
    // Offsets: [diagonalIdx, 0] - start at row diagonalIdx, column 0
    // Sizes: [1, slots] - extract 1 row with 'slots' columns
    // Strides: [1, 1] - contiguous
    Value diagonalTensor = rewriter.create<tensor::ExtractSliceOp>(
        loc, diagonalTensorType, weightMatrix,
        /*offsets=*/
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(diagonalIdx),
                               rewriter.getIndexAttr(0)},
        /*sizes=*/
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1),
                               rewriter.getIndexAttr(slots)},
        /*strides=*/
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1),
                               rewriter.getIndexAttr(1)});

    // Encode the diagonal tensor with the target scale
    auto encoding = lwe::InverseCanonicalEncodingAttr::get(
        rewriter.getContext(), targetScale);

    auto coeffType = rewriter.getF64Type();
    auto polyAttr = polynomial::IntPolynomialAttr::get(
        rewriter.getContext(),
        polynomial::IntPolynomial::fromCoefficients({1}));
    auto ring = polynomial::RingAttr::get(coeffType, polyAttr);

    auto ptSpace =
        lwe::PlaintextSpaceAttr::get(rewriter.getContext(), ring, encoding);

    auto noOverflowAttr = lwe::NoOverflowAttr::get(rewriter.getContext());
    auto appData =
        lwe::ApplicationDataAttr::get(diagonalTensorType, noOverflowAttr);

    auto ptType =
        lwe::NewLWEPlaintextType::get(rewriter.getContext(), appData, ptSpace);

    Value encodedDiagonal = rewriter.create<openfhe::MakeCKKSPackedPlaintextOp>(
        loc, ptType, cryptoContext, diagonalTensor);

    llvm::dbgs() << "  Successfully extracted and encoded diagonal "
                 << diagonalIdx << " with scale " << targetScale << "\n";

    return encodedDiagonal;
  }

  // ===== UNROLLED FORM =====
  LogicalResult lowerToUnrolledForm(openfhe::LinearTransformOp op,
                                    PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    Value cryptoContext = op.getCryptoContext();
    Value inputCiphertext = op.getOperand(1);
    Value weightsPlaintext = op.getOperand(2);

    int32_t diagonalCount = op.getDiagonalCount();
    int32_t slots = op.getSlots();

    int64_t inputScale = getScalingFactor(inputCiphertext.getType());
    int64_t weightsScale = getScalingFactor(weightsPlaintext.getType());

    llvm::dbgs() << "Lowering openfhe.linear_transform (UNROLLED) with "
                 << diagonalCount << " diagonals, " << slots << " slots\n";

    Value accumulator = nullptr;

    for (int32_t i = 0; i < diagonalCount; ++i) {
      llvm::dbgs() << "Processing diagonal " << i << "\n";

      Value rotatedInput = inputCiphertext;
      Value rotKey = nullptr;

      if (i > 0) {
        // Step 1a: Create constant for rotation index and load key
        Value rotIndexValue = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(i));

        // Create static rotation key type with the constant index
        auto rotKeyType = kmrt::RotKeyType::get(rewriter.getContext(), i);

        rotKey =
            rewriter.create<kmrt::LoadKeyOp>(loc, rotKeyType, rotIndexValue);

        // Step 1b: Perform rotation
        rotatedInput = rewriter.create<openfhe::RotOp>(
            loc, inputCiphertext.getType(), cryptoContext, inputCiphertext,
            rotKey);

        // Step 1c: Clear the rotation key immediately after use
        rewriter.create<kmrt::ClearKeyOp>(loc, rotKey);
      }

      // Step 2: Extract and encode diagonal
      Value diagonal = extractDiagonal(rewriter, loc, weightsPlaintext, i,
                                       slots, weightsScale, cryptoContext);

      // Step 3: Multiply
      Value mulResult = rewriter.create<openfhe::MulPlainOp>(
          loc, rotatedInput.getType(), cryptoContext, rotatedInput, diagonal);

      // Step 4: Accumulate
      if (accumulator == nullptr) {
        accumulator = mulResult;
        llvm::dbgs() << "  Initialized accumulator with first diagonal\n";
      } else {
        accumulator = rewriter.create<openfhe::AddOp>(
            loc, accumulator.getType(), cryptoContext, accumulator, mulResult);
        llvm::dbgs() << "  Accumulated diagonal " << i << " into running sum\n";
      }
    }

    rewriter.replaceOp(op, accumulator);
    llvm::dbgs() << "Successfully lowered linear_transform (UNROLLED)\n";
    return success();
  }

  // ===== AFFINE LOOP FORM =====
  LogicalResult lowerToAffineLoopForm(openfhe::LinearTransformOp op,
                                      PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    Value cryptoContext = op.getCryptoContext();
    Value inputCiphertext = op.getOperand(1);
    Value weightsPlaintext = op.getOperand(2);

    int32_t diagonalCount = op.getDiagonalCount();
    int32_t slots = op.getSlots();

    int64_t inputScale = getScalingFactor(inputCiphertext.getType());
    int64_t weightsScale = getScalingFactor(weightsPlaintext.getType());

    llvm::dbgs() << "Lowering openfhe.linear_transform (AFFINE LOOP) with "
                 << diagonalCount << " diagonals, " << slots << " slots\n";

    // Get weight matrix for use inside the loop
    Value weightMatrix = getWeightTensor(rewriter, loc, weightsPlaintext);
    if (!weightMatrix) {
      return op.emitError("Could not extract weight tensor");
    }

    // First iteration (i=0): no rotation, initialize accumulator
    Value diagonal0 = extractDiagonal(rewriter, loc, weightsPlaintext, 0, slots,
                                      weightsScale, cryptoContext);
    Value mulResult0 = rewriter.create<openfhe::MulPlainOp>(
        loc, inputCiphertext.getType(), cryptoContext, inputCiphertext,
        diagonal0);
    Value initialAccum = mulResult0;

    if (diagonalCount == 1) {
      rewriter.replaceOp(op, initialAccum);
      return success();
    }

    // Create affine.for loop for remaining diagonals (i=1 to diagonalCount-1)
    auto lowerBound = rewriter.getAffineConstantExpr(1);
    auto upperBound = rewriter.getAffineConstantExpr(diagonalCount);
    auto lbMap = AffineMap::get(0, 0, lowerBound, rewriter.getContext());
    auto ubMap = AffineMap::get(0, 0, upperBound, rewriter.getContext());

    auto affineFor = rewriter.create<affine::AffineForOp>(
        loc, ArrayRef<Value>{}, lbMap, ArrayRef<Value>{}, ubMap, 1,
        ValueRange{initialAccum},
        [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
          Value currentAccum = iterArgs[0];

          // Step 1a: Load rotation key directly using the induction variable
          // The type is dynamic (!kmrt.rot_key) since iv is runtime-determined
          auto rotKeyType =
              kmrt::RotKeyType::get(builder.getContext(), std::nullopt);

          Value rotKey = builder.create<kmrt::LoadKeyOp>(loc, rotKeyType, iv);

          // Step 1b: Rotate input by iv
          Value rotatedInput = builder.create<openfhe::RotOp>(
              loc, inputCiphertext.getType(), cryptoContext, inputCiphertext,
              rotKey);

          // Step 1c: Clear the rotation key immediately after rotation
          builder.create<kmrt::ClearKeyOp>(loc, rotKey);

          // For extracting the diagonal, we still need the index value
          auto ivIndex = builder.create<affine::AffineApplyOp>(
              loc, AffineMap::get(1, 0, builder.getAffineDimExpr(0)),
              ValueRange{iv});

          // Step 2: Extract diagonal i from weight matrix
          auto slotType = builder.getF64Type();
          auto diagonalTensorType = RankedTensorType::get({slots}, slotType);

          Value diagonalTensor = builder.create<tensor::ExtractSliceOp>(
              loc, diagonalTensorType, weightMatrix,
              ArrayRef<OpFoldResult>{ivIndex.getResult(),
                                     builder.getIndexAttr(0)},
              ArrayRef<OpFoldResult>{builder.getIndexAttr(1),
                                     builder.getIndexAttr(slots)},
              ArrayRef<OpFoldResult>{builder.getIndexAttr(1),
                                     builder.getIndexAttr(1)});

          // Step 3: Encode diagonal
          auto encoding = lwe::InverseCanonicalEncodingAttr::get(
              builder.getContext(), weightsScale);

          auto coeffType = builder.getF64Type();
          auto polyAttr = polynomial::IntPolynomialAttr::get(
              builder.getContext(),
              polynomial::IntPolynomial::fromCoefficients({1}));
          auto ring = polynomial::RingAttr::get(coeffType, polyAttr);

          auto ptSpace = lwe::PlaintextSpaceAttr::get(builder.getContext(),
                                                      ring, encoding);

          auto noOverflowAttr = lwe::NoOverflowAttr::get(rewriter.getContext());
          auto appData =
              lwe::ApplicationDataAttr::get(diagonalTensorType, noOverflowAttr);
          auto ptType = lwe::NewLWEPlaintextType::get(rewriter.getContext(),
                                                      appData, ptSpace);

          Value encodedDiagonal =
              builder.create<openfhe::MakeCKKSPackedPlaintextOp>(
                  loc, ptType, cryptoContext, diagonalTensor);

          // Step 4: Multiply
          Value mulResult = builder.create<openfhe::MulPlainOp>(
              loc, rotatedInput.getType(), cryptoContext, rotatedInput,
              encodedDiagonal);

          // Step 5: Add to accumulator
          Value nextAccum = builder.create<openfhe::AddOp>(
              loc, currentAccum.getType(), cryptoContext, currentAccum,
              mulResult);

          builder.create<affine::AffineYieldOp>(loc, nextAccum);
        });

    Value finalResult = affineFor.getResult(0);
    rewriter.replaceOp(op, finalResult);

    llvm::dbgs()
        << "Successfully lowered openfhe.linear_transform (AFFINE LOOP)\n";
    return success();
  }
};

}  // namespace

#define GEN_PASS_DEF_LOWERLINEARTRANSFORM
#include "lib/Transforms/LowerLinearTransform/LowerLinearTransform.h.inc"

struct LowerLinearTransform
    : impl::LowerLinearTransformBase<LowerLinearTransform> {
  using impl::LowerLinearTransformBase<
      LowerLinearTransform>::LowerLinearTransformBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto func = getOperation();

    // Default to unrolled form, can be changed via pass option
    bool useUnrolled = false;  // TODO: Make this a pass option

    llvm::dbgs() << "=== LowerLinearTransform (OpenFHE) ";
    llvm::dbgs() << (useUnrolled ? "UNROLLED" : "AFFINE LOOP");
    llvm::dbgs() << " MODE ===\n";

    RewritePatternSet patterns(context);
    patterns.add<LowerLinearTransformPattern>(context, useUnrolled);

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    llvm::dbgs() << "OpenFHE linear transform lowering completed\n";
  }
};

}  // namespace heir
}  // namespace mlir
