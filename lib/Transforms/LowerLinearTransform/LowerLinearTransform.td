#ifndef LIB_TRANSFORMS_LOWERLINEARTRANSFORM_LOWERLINEARTRANSFORM_TD_
#define LIB_TRANSFORMS_LOWERLINEARTRANSFORM_LOWERLINEARTRANSFORM_TD_

include "mlir/Pass/PassBase.td"

def LowerLinearTransform : Pass<"lower-linear-transform", "func::FuncOp"> {
  let summary = "Lower CKKS linear transform operations to explicit FHE operations";
  let description = [{
    This pass lowers high-level ckks.linear_transform operations into explicit
    sequences of rotation, multiplication, and addition operations.

    The naive lowering strategy processes each diagonal independently:
    1. Extract diagonal from weights tensor
    2. Apply rotation to input ciphertext (if needed)
    3. Multiply rotated ciphertext with diagonal plaintext
    4. Accumulate all partial results

    Example transformation:
    ```mlir
    // Before lowering:
    %result = ckks.linear_transform %input, %weights {
      diagonal_count = 4, slots = 8
    } : (!ct, !pt) -> !ct

    // After lowering:
    %diag0 = extract_diagonal %weights, 0
    %mul0 = ckks.mul_plain %input, %diag0 : (!ct, !pt) -> !ct

    %rot1 = ckks.rotate %input {offset = 1} : !ct
    %diag1 = extract_diagonal %weights, 1
    %mul1 = ckks.mul_plain %rot1, %diag1 : (!ct, !pt) -> !ct

    %rot2 = ckks.rotate %input {offset = 2} : !ct
    %diag2 = extract_diagonal %weights, 2
    %mul2 = ckks.mul_plain %rot2, %diag2 : (!ct, !pt) -> !ct

    %rot3 = ckks.rotate %input {offset = 3} : !ct
    %diag3 = extract_diagonal %weights, 3
    %mul3 = ckks.mul_plain %rot3, %diag3 : (!ct, !pt) -> !ct

    %acc1 = ckks.add %mul0, %mul1 : (!ct, !ct) -> !ct
    %acc2 = ckks.add %mul2, %mul3 : (!ct, !ct) -> !ct
    %result = ckks.add %acc1, %acc2 : (!ct, !ct) -> !ct
    ```
  }];

  let dependentDialects = [
    "mlir::heir::openfhe::OpenfheDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
    "mlir::affine::AffineDialect"
  ];
}
#endif  // LIB_TRANSFORMS_LOWERLINEARTRANSFORM_LOWERLINEARTRANSFORM_TD_
