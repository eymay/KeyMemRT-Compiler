#ifndef LIB_DIALECT_LWE_IR_LWEOPS_TD_
#define LIB_DIALECT_LWE_IR_LWEOPS_TD_

include "lib/Dialect/LWE/IR/LWEDialect.td"
include "lib/Dialect/LWE/IR/LWETraits.td"
include "lib/Dialect/LWE/IR/LWETypes.td"
include "lib/Dialect/Polynomial/IR/PolynomialAttributes.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"

class HasEncoding<
  string encodingHolder,
  string encoding,
  string ty,
  string comparator = "std::equal_to<>()"
> : PredOpTrait<
    "the first arg's type's encoding matches the given encoding",
    CPred<
      comparator # "(" #
      "::llvm::cast<lwe::" # ty # ">($" # encodingHolder # ".getType()).getPlaintextSpace().getEncoding(), " #
      "$" # encoding # ")"
    >
  >;


class EncodingsMatch<
  string encodingHolder1,
  string ty1,
  string encodingHolder2,
  string ty2,
  string comparator = "std::equal_to<>()"
> : PredOpTrait<
    "the first arg's type's encoding matches the given encoding",
    CPred<
      comparator # "(" #
      "::llvm::cast<lwe::" # ty1 # ">($" # encodingHolder1 # ".getType()).getPlaintextSpace().getEncoding(), " #
      "::llvm::cast<lwe::" # ty2 # ">($" # encodingHolder2 # ".getType()).getPlaintextSpace().getEncoding())"
    >
  >;

class KeyAndCiphertextMatch<
  string v1, string ty1, string v2, string ty2,
  string comparator = "std::equal_to<>()"
> : PredOpTrait<
    "the first value's RLWE parameters matches the second value's",
    CPred<
      comparator # "(" #
      "::llvm::cast<lwe::" # ty1 # ">($" # v1 # ".getType()).getRing(), " #
      "::llvm::cast<lwe::" # ty2 # ">($" # v2 # ".getType()).getCiphertextSpace().getRing())"
    >
  >;


// LWE Operations are always Pure by design
class LWE_Op<string mnemonic, list<Trait> traits = []> :
        Op<LWE_Dialect, mnemonic,  traits # [Pure]> {
  let cppNamespace = "::mlir::heir::lwe";
  let assemblyFormat = [{
    operands attr-dict `:`  functional-type(operands, results)
  }];
}

class LWE_BinOp<string mnemonic, list<Trait> traits = []> :
        LWE_Op<mnemonic,  traits # [ElementwiseMappable]> {
  let arguments = (ins LWECiphertext:$lhs, LWECiphertext:$rhs);
  let results = (outs LWECiphertext:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

def LWE_RAddOp : LWE_BinOp<"radd", [SameOperandsAndResultRings, Commutative]> {
  let summary = "Add two RLWE ciphertexts";
  let assemblyFormat = [{
    operands attr-dict `:`  functional-type(operands, results)
  }];
}

def LWE_RSubOp : LWE_BinOp<"rsub", [SameOperandsAndResultRings]> {
  let summary = "Subtract two RLWE ciphertexts";
  let assemblyFormat = [{
    operands attr-dict `:`  functional-type(operands, results)
  }];
}

def LWE_RMulOp : LWE_BinOp<"rmul", [SameOperandsAndResultRings, Commutative]> {
  let summary = "Multiplies two RLWE ciphertexts";
  let assemblyFormat = [{
    operands attr-dict `:`  functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def LWE_RNegateOp : LWE_Op<"rnegate", [SameOperandsAndResultType, ElementwiseMappable]> {
  let summary = "Negate a RLWE ciphertexts";
  let arguments = (ins LWECiphertextLike:$input);
  let results = (outs LWECiphertextLike:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

def LWE_RLWEEncodeOp : LWE_Op<"rlwe_encode", [HasEncoding<"output", "encoding", "LWEPlaintextType">]> {
  let summary = "Encode an integer to yield an RLWE plaintext";
  let description = [{
    Encode an integer to yield an RLWE plaintext.

    This op uses a an encoding attribute to encode the bits of the integer into
    an RLWE plaintext value that can then be encrypted. CKKS cleartext inputs may
    be floating points, and a scaling factor described by the encoding will be
    applied.

    Examples:

    ```
    %Y = lwe.rlwe_encode %value {encoding = #enc, ring = #ring}: i1 to !lwe.rlwe_plaintext<encoding = #enc, ring = #ring>
    ```
  }];

  let arguments = (ins
    SignlessIntegerOrFloatLike:$input,
    AnyPlaintextEncodingAttr:$encoding,
    Polynomial_RingAttr:$ring
  );

  let results = (outs LWEPlaintext:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` qualified(type($output))";

  // Verify that the input type and the encoding are compatible.
  let hasVerifier = true;
}

def LWE_RLWEDecodeOp : LWE_Op<"rlwe_decode", [HasEncoding<"input", "encoding", "LWEPlaintextType">]> {
  let summary = "Decode an RLWE plaintext to an underlying type";

  let arguments = (ins
    LWEPlaintext:$input,
    AnyPlaintextEncodingAttr:$encoding,
    Polynomial_RingAttr:$ring
  );

  let results = (outs SignlessIntegerOrFloatLike:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` qualified(type($output))";

  // Verify that the expected output type and the encoding are compatible.
  let hasVerifier = true;
}

def LWE_RLWEEncryptOp : LWE_Op<"rlwe_encrypt", [
    EncodingsMatch<"input", "LWEPlaintextType", "output", "LWECiphertextType">]> {
  let summary = "Encrypt an RLWE plaintext to a RLWE ciphertext";
  let description = [{
    Encrypt an RLWE plaintext to yield a RLWE ciphertext.
  }];

  let arguments = (ins
    LWEPlaintext:$input,
    LWESecretOrPublicKey:$key
  );
  let results = (outs LWECiphertext:$output);
  let hasVerifier = 1;
}

def LWE_RLWEDecryptOp : LWE_Op<"rlwe_decrypt", [
    EncodingsMatch<"input", "LWECiphertextType", "output", "LWEPlaintextType">,
    KeyAndCiphertextMatch<"secret_key", "LWESecretKeyType", "input", "LWECiphertextType">]> {
  let summary = "Decrypt an RLWE ciphertext to a RLWE plaintext";
  let description = [{
    Decrypt an RLWE ciphertext to yield a RLWE plaintext
  }];

  let arguments = (ins
    LWECiphertext:$input,
    LWESecretKey:$secret_key
  );
  let results = (outs LWEPlaintext:$output);
}

def LWE_LinearTransformOp : LWE_Op<"linear_transform"> {
  let summary = "Linear transformation operation using precomputed diagonal plaintexts";
  let description = [{
    This operation performs a linear transformation using precomputed diagonal plaintexts.
    It's commonly used in homomorphic encryption to implement matrix-vector multiplication
    or convolution operations efficiently using the diagonal method and rotations.
    
    The operation takes a ciphertext input and plaintext weights (diagonals) as inputs,
    and produces a ciphertext result representing the linear transformation.
    
    Attributes:
    - diagonal_count: Number of diagonals in the weight matrix
    - slots: Number of slots in the ciphertext
  }];

  let arguments = (ins
    LWECiphertext:$input,
    LWEPlaintext:$weights,
    I32Attr:$diagonal_count,
    I32Attr:$slots
  );

  let results = (outs
    LWECiphertext:$result
  );

  let assemblyFormat = [{
    $input `,` $weights 
    attr-dict `:` `(` qualified(type($input)) `,` qualified(type($weights)) `)` 
    `->` qualified(type($result))
  }];
}

#endif  // LIB_DIALECT_LWE_IR_LWEOPS_TD_
