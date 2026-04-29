#ifndef LIB_DIALECT_ORION_IR_ORIONOPS_TD_
#define LIB_DIALECT_ORION_IR_ORIONOPS_TD_

include "lib/Dialect/Orion/IR/OrionDialect.td"
include "lib/Dialect/LWE/IR/LWETypes.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Orion_Op<string mnemonic, list<Trait> traits = []> :
        Op<Orion_Dialect, mnemonic, traits # [Pure]> {
  let cppNamespace = "::mlir::heir::orion";
}

def Orion_LinearTransformOp : Orion_Op<"linear_transform"> {
  let summary = "Linear transformation operation using precomputed diagonal plaintexts";
  let description = [{
    This operation performs a linear transformation using precomputed diagonal
    plaintexts. It's commonly used in homomorphic encryption to implement
    matrix-vector multiplication or convolution operations efficiently using
    the diagonal method and rotations.

    The operation takes a ciphertext input and a tensor of weights (diagonals)
    and produces a ciphertext result representing the linear transformation.

    Example:
    ```mlir
    %result = orion.linear_transform %input, %weights {
      block_row = 0 : i32,
      block_col = 0 : i32,
      diagonal_count = 128 : i32,
      slots = 4096 : i32,
      bsgs_ratio = 2.0 : f64,
      orion_level = 5 : i32
    } : (!ct, tensor<128x4096xf64>) -> !ct
    ```
  }];

  let arguments = (ins
    LWECiphertext:$input,
    AnyRankedTensor:$weights,

    // Block matrix coordinates
    DefaultValuedAttr<I32Attr, "0">:$block_row,
    DefaultValuedAttr<I32Attr, "0">:$block_col,

    // Diagonal parameters
    DefaultValuedAttr<I32Attr, "128">:$diagonal_count,
    DefaultValuedAttr<I32Attr, "4096">:$slots,

    // Optional indices for the represented diagonals (translator emits this)
    OptionalAttr<DenseI32ArrayAttr>:$diagonal_indices,

    // Optimization parameters
    DefaultValuedAttr<F64Attr, "2.0">:$bsgs_ratio,
    DefaultValuedAttr<I32Attr, "5">:$orion_level,

    // Optional matrix dimensions
    OptionalAttr<I32Attr>:$matrix_rows,
    OptionalAttr<I32Attr>:$matrix_cols,

    // Optional BSGS parameters
    OptionalAttr<I32Attr>:$baby_step_size,
    OptionalAttr<I32Attr>:$giant_step_size
  );

  let results = (outs
    LWECiphertext:$output
  );

  let assemblyFormat = "$input `,` $weights attr-dict `:` `(` type($input) `,` qualified(type($weights)) `)` `->` type($output)" ;
}

def Orion_ChebyshevOp : Orion_Op<"chebyshev"> {
  let summary = "Evaluate a Chebyshev polynomial series on a ciphertext.";

  let description = [{
    Evaluates a Chebyshev polynomial series on a ciphertext using pre-computed
    coefficients. This operation directly maps to OpenFHE's EvalChebyshevSeries.

    The coefficients correspond to the Chebyshev series:
    f(x) = c_0 T_0(x) + c_1 T_1(x) + c_2 T_2(x) + ... + c_n T_n(x)

    where T_i(x) are Chebyshev polynomials of the first kind.

    Example:
    ```mlir
    %result = orion.chebyshev %input {
        coefficients = [1.0, 0.0, -0.33333, 0.0, 0.2],
        domain_start = -1.0,
        domain_end = 1.0
    } : (!lwe.lwe_ciphertext) -> !lwe.lwe_ciphertext
    ```
  }];

  let arguments = (ins
    LWECiphertext:$input,
    F64ArrayAttr:$coefficients,
    DefaultValuedAttr<F64Attr, "-1.0">:$domain_start,
    DefaultValuedAttr<F64Attr, "1.0">:$domain_end
  );

  let results = (outs
    LWECiphertext:$output
  );

  let assemblyFormat = [{
    operands attr-dict `:`  functional-type(operands, results)
  }];
}

#endif  // LIB_DIALECT_ORION_IR_ORIONOPS_TD_
