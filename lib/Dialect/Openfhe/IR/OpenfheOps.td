#ifndef LIB_DIALECT_OPENFHE_IR_OPENFHEOPS_TD_
#define LIB_DIALECT_OPENFHE_IR_OPENFHEOPS_TD_

include "OpenfheDialect.td"
include "OpenfheTypes.td"

include "lib/Dialect/LWE/IR/LWETypes.td"
include "lib/Dialect/LWE/IR/LWETraits.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Openfhe_Op<string mnemonic, list<Trait> traits = []> :
        Op<Openfhe_Dialect, mnemonic, traits> {
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let cppNamespace = "::mlir::heir::openfhe";
}

class Openfhe_UnaryTypeSwitchOp<string mnemonic, list<Trait> traits = []>
  : Openfhe_Op<mnemonic, traits # [
    Pure,
]>{
 let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWECiphertext:$ciphertext
  );
  let results = (outs NewLWECiphertext:$output);
}

class Openfhe_UnaryOp<string mnemonic, list<Trait> traits = []>
  : Openfhe_UnaryTypeSwitchOp<
      mnemonic, traits # [AllTypesMatch<["ciphertext", "output"]>]
    >;

class Openfhe_BinaryOp<string mnemonic, list<Trait> traits = []>
  :  Openfhe_Op<mnemonic, traits # [
    Pure
]>{
 let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWECiphertext:$lhs,
    NewLWECiphertext:$rhs
  );
  let results = (outs NewLWECiphertext:$output);
}

class Openfhe_BinaryInPlaceOp<string mnemonic, list<Trait> traits = []>
  : Openfhe_Op<mnemonic, traits # [
    AllTypesMatch<["lhs", "rhs"]>,
    ]> {

  let summary = "In-place binary operation for OpenFHE";

  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWECiphertext:$lhs,
    NewLWECiphertext:$rhs
  );
}


def GenParamsOp : Openfhe_Op<"gen_params"> {
  let description = [{
    Generates the parameters for the OpenFHE scheme.

    `mulDepth` is the depth of the multiplication circuit,
    including the bootstrapping depth.

    `plainMod` is the modulus of the plaintext space. If we
    are using CKKS, this is 0.

    `insecure` is a flag that determines whether the parameters
    are generated securely or not. In Openfhe, this means setting
    HEStd_NotSet for security level.

    For other flags, see the OpenFHE documentation in
    https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/README.md#description-of-the-cryptocontext-parameters-and-their-restrictions
  }];
  let arguments = (ins
    // Essential parameters
    I64Attr:$mulDepth,
    I64Attr:$plainMod,
    // Optional parameters
    DefaultValuedAttr<I64Attr, "0">:$ringDim,
    DefaultValuedAttr<I64Attr, "0">:$batchSize,
    // Modulus chain parameters
    DefaultValuedAttr<I64Attr, "0">:$firstModSize,
    DefaultValuedAttr<I64Attr, "0">:$scalingModSize,
    // KPZ21 way of noise estimation
    DefaultValuedAttr<I64Attr, "0">:$evalAddCount,
    DefaultValuedAttr<I64Attr, "0">:$keySwitchCount,
    // Key switching technique parameters
    DefaultValuedAttr<I64Attr, "0">:$digitSize,
    DefaultValuedAttr<I64Attr, "0">:$numLargeDigits,
    // Relinearization parameters
    DefaultValuedAttr<I64Attr, "0">:$maxRelinSkDeg,
    // Option switches
    DefaultValuedAttr<BoolAttr, "false">:$insecure,
    DefaultValuedAttr<BoolAttr, "false">:$encryptionTechniqueExtended,
    DefaultValuedAttr<BoolAttr, "false">:$keySwitchingTechniqueBV,
    DefaultValuedAttr<BoolAttr, "false">:$scalingTechniqueFixedManual
  );
  let results = (outs Openfhe_CCParams:$params);
}

def GenContextOp : Openfhe_Op<"gen_context"> {
  let arguments = (ins
    Openfhe_CCParams:$params,
    BoolAttr:$supportFHE
  );
  let results = (outs Openfhe_CryptoContext:$context);
}

def GenMulKeyOp : Openfhe_Op<"gen_mulkey"> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_PrivateKey:$privateKey
  );
}

def GenRotKeyOp : Openfhe_Op<"gen_rotkey"> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_PrivateKey:$privateKey,
    DenseI64ArrayAttr:$indices
  );
}

def GenRotKeyDepthOp : Openfhe_Op<"gen_rotkey_depth"> {
  let summary = "Generate rotation keys at a specific depth";
  let description = [{
    This operation generates rotation keys for the specified indices at a given depth.
    It allows for efficient key generation where keys are only created at depths where they are needed.
  }];
  
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_PrivateKey:$privateKey,
    ArrayAttr:$indices,
    I64Attr:$depth
  );
}

def SetupBootstrapOp : Openfhe_Op<"setup_bootstrap"> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Builtin_IntegerAttr:$levelBudgetEncode,
    Builtin_IntegerAttr:$levelBudgetDecode
  );
}

def GenBootstrapKeyOp : Openfhe_Op<"gen_bootstrapkey"> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_PrivateKey:$privateKey
  );
}

def MakePackedPlaintextOp : Openfhe_Op<"make_packed_plaintext", [Pure]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RankedTensorOf<[AnyInteger]>:$value)
  ;
  let results = (outs NewLWEPlaintext:$plaintext);
  let hasVerifier = 1;
}

def MakeCKKSPackedPlaintextOp : Openfhe_Op<"make_ckks_packed_plaintext", [Pure]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    RankedTensorOf<[AnyFloat, AnyInteger]>:$value)
  ;
  let results = (outs NewLWEPlaintext:$plaintext);
  let hasVerifier = 1;
}

def EncryptOp : Openfhe_Op<"encrypt", [Pure]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWEPlaintext:$plaintext,
    Openfhe_PublicKeyOrPrivateKey:$encryptionKey)
  ;
  let results = (outs NewLWECiphertext:$ciphertext);
}

def DecryptOp : Openfhe_Op<"decrypt", [Pure]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWECiphertext:$ciphertext,
    Openfhe_PrivateKey:$privateKey)
  ;
  let results = (outs NewLWEPlaintext:$plaintext);
}

def AddOp : Openfhe_BinaryOp<"add",
    [SameOperandsAndResultRings,
    InferTypeOpAdaptor]> {
  let summary = "OpenFHE add operation of two ciphertexts.";
}
def SubOp : Openfhe_BinaryOp<"sub",
    [SameOperandsAndResultRings,
    InferTypeOpAdaptor]> {
  let summary = "OpenFHE sub operation of two ciphertexts.";
}

// In-Place Addition
def AddInPlaceOp : Openfhe_BinaryInPlaceOp<"add_inplace"> {
  let summary = "Performs in-place homomorphic addition,  modifying lhs.";
}

// In-Place Subtraction
def SubInPlaceOp : Openfhe_BinaryInPlaceOp<"sub_inplace"> {
  let summary = "Performs in-place homomorphic subtraction, modifying lhs.";
}


def AddPlainOp : Openfhe_Op<"add_plain",[
    Pure,
    AllCiphertextTypesMatch,
    InferTypeOpAdaptor
]> {
  let summary = "OpenFHE add operation of a ciphertext and a plaintext.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWEPlaintextOrCiphertext:$lhs,
    NewLWEPlaintextOrCiphertext:$rhs
  );
  let results = (outs NewLWECiphertext:$output);
}

def SubPlainOp : Openfhe_Op<"sub_plain",[
    Pure,
    AllCiphertextTypesMatch,
    InferTypeOpAdaptor
]> {
  let summary = "OpenFHE sub operation of a ciphertext and a plaintext.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWEPlaintextOrCiphertext:$lhs,
    NewLWEPlaintextOrCiphertext:$rhs
  );
  let results = (outs NewLWECiphertext:$output);
}

def MulOp : Openfhe_BinaryOp<"mul"> { let summary = "OpenFHE mul operation of two ciphertexts with relinearization."; }

def MulNoRelinOp : Openfhe_Op<"mul_no_relin", [Pure, SameOperandsAndResultRings, InferTypeOpAdaptor]> {
  let summary = "OpenFHE mul operation of two ciphertexts without relinearization.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWECiphertext:$lhs,
    NewLWECiphertext:$rhs
  );
  let results = (outs NewLWECiphertext:$output);
  let hasVerifier = 1;
}

def MulPlainOp : Openfhe_Op<"mul_plain",[
    Pure
]> {
  let summary = "OpenFHE mul operation of a ciphertext and a plaintext.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWECiphertext:$ciphertext,
    NewLWEPlaintext:$plaintext
  );
  let results = (outs NewLWECiphertext:$output);
}

def MulConstOp : Openfhe_Op<"mul_const",[
    Pure,
    AllTypesMatch<["ciphertext", "output"]>
]> {
  let summary = "OpenFHE mul operation of a ciphertext and a constant.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWECiphertext:$ciphertext,
    I64:$constant
  );
  let results = (outs NewLWECiphertext:$output);
}

def NegateOp : Openfhe_UnaryOp<"negate"> { let summary = "OpenFHE negate operation of a ciphertext."; }
def SquareOp : Openfhe_UnaryOp<"square"> { let summary = "OpenFHE square operation of a ciphertext."; }
def RelinOp : Openfhe_UnaryTypeSwitchOp<"relin"> { let summary = "OpenFHE relinearize operation of a ciphertext."; }

def ModReduceOp : Openfhe_UnaryTypeSwitchOp<"mod_reduce"> { let summary = "OpenFHE mod_reduce operation of a ciphertext. (used only for BGV/CKKS)"; }
def LevelReduceOp : Openfhe_UnaryTypeSwitchOp<"level_reduce"> {
  let summary = "OpenFHE level_reduce operation of a ciphertext.";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWECiphertext:$ciphertext,
    DefaultValuedAttr<I64Attr, "1">:$levelToDrop
  );
}

def RotOp : Openfhe_Op<"rot",[
  Pure,
  AllTypesMatch<["ciphertext", "output"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWECiphertext:$ciphertext,
    Openfhe_EvalKey:$evalKey
  );
  let results = (outs NewLWECiphertext:$output);
}

def AutomorphOp : Openfhe_Op<"automorph", [
  Pure,
  AllTypesMatch<["ciphertext", "output"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWECiphertext:$ciphertext,
    Openfhe_EvalKey:$evalKey
  );
  let results = (outs NewLWECiphertext:$output);
}

def KeySwitchOp : Openfhe_Op<"key_switch", [
   Pure,
   AllTypesMatch<["ciphertext", "output"]>
]> {
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWECiphertext:$ciphertext,
    Openfhe_EvalKey:$evalKey
  );
  let results = (outs NewLWECiphertext:$output);
}

def BootstrapOp : Openfhe_UnaryTypeSwitchOp<"bootstrap"> { let summary = "OpenFHE bootstrap operation of a ciphertext. (For CKKS)"; }

def DeserializeKeyOp : Openfhe_Op<"deserialize_key"> {
  let summary = "Deserialize a previously serialized rotation key";
  let description = [{
    Takes a serialized evaluation key and deserializes it back into the context.
    This operation just loads the key from a file into context memory.

    Example:
    ```mlir
    openfhe.deserialize_key %cc, %ek : !openfhe.crypto_context, !openfhe.eval_key
    ```
  }];

  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Builtin_IntegerAttr:$index
  );
  let results = (outs Openfhe_EvalKey:$evalKey);
}

def SerializeKeyOp : Openfhe_Op<"serialize_key"> {
  let summary = "Serialize a rotation key for a given rotation index";
  let description = [{
    Serializes a rotation key for a given rotation index, returning an evaluation key
    that encapsulates both the key data and the rotation amount.

    Example:
    ```mlir
    %ek = openfhe.serialize_key %cc, 1 : !openfhe.crypto_context -> !openfhe.eval_key
    %rotated = openfhe.rot %cc, %ct, %ek : !openfhe.crypto_context, !openfhe.lwe_ciphertext, !openfhe.eval_key -> !openfhe.lwe_ciphertext
    ```
  }];
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_EvalKey:$evalKey
  );
}

def ClearKeyOp : Openfhe_Op<"clear_key"> {
  let summary = "Clear a rotation key from memory";
  let description = [{
    Takes an evaluation key and clears it from the context memory.
    This helps manage memory usage by removing keys when they're no longer needed.

    Example:
    ```mlir
    openfhe.clear_key %cc, %ek : !openfhe.crypto_context, !openfhe.eval_key
    ```
  }];

  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_EvalKey:$evalKey
  );
}

def CompressKeyOp : Openfhe_Op<"compress_key"> {
  let summary = "Compress a rotation key to reduce memory usage";
  let description = [{
    Takes an evaluation key and compresses it to a specified depth to reduce memory usage, 
    returning the compressed key which can still be used in operations. The compressed key 
    maintains the same rotation index as the original key but with reduced depth.
    
    Example:
    ```mlir
    %compressed_ek = openfhe.compress_key %cc, %ek { depth = 5 } : !openfhe.crypto_context, !openfhe.eval_key -> !openfhe.eval_key
    ```
  }];
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    Openfhe_EvalKey:$evalKey,
    I32Attr:$depth
  );
  let results = (outs Openfhe_EvalKey:$resultKey);
  
  let assemblyFormat = [{
    $cryptoContext `,` $evalKey attr-dict `:` type($cryptoContext) `,` type($evalKey) `->` type($resultKey)
  }];
}

// In-Place Plaintext Operations
def AddPlainInPlaceOp : Openfhe_Op<"add_plain_inplace"> {
  let summary = "In-place addition of ciphertext and plaintext";
  let description = [{
    Performs in-place homomorphic addition of a ciphertext and plaintext,
    modifying the ciphertext operand in place.
  }];
  
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWECiphertext:$ciphertext,
    NewLWEPlaintext:$plaintext
  );
}

def SubPlainInPlaceOp : Openfhe_Op<"sub_plain_inplace"> {
  let summary = "In-place subtraction of plaintext from ciphertext";
  let description = [{
    Performs in-place homomorphic subtraction (ciphertext - plaintext),
    modifying the ciphertext operand in place.
  }];
  
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWECiphertext:$ciphertext,
    NewLWEPlaintext:$plaintext
  );
}


// In-Place Constant Multiplication (this definitely exists)
def MulConstInPlaceOp : Openfhe_Op<"mul_const_inplace"> {
  let summary = "In-place multiplication of ciphertext and integer constant";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWECiphertext:$ciphertext,
    I64:$constant
  );
}

// In-Place Unary Operations (these definitely exist)
def NegateInPlaceOp : Openfhe_Op<"negate_inplace"> {
  let summary = "In-place negation of a ciphertext";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWECiphertext:$ciphertext
  );
}

def RelinInPlaceOp : Openfhe_Op<"relin_inplace"> {
  let summary = "In-place relinearization of a ciphertext";
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWECiphertext:$ciphertext
  );
}

// In-place rotation operation (self-assignment style)
def RotInPlaceOp : Openfhe_Op<"rot_inplace"> {
  let summary = "In-place rotation operation that modifies the input ciphertext";
  let description = [{
    Performs rotation in-place, modifying the input ciphertext. This operation
    is used to model self-assignment in C++ code generation:
    `ciphertext = cc->EvalRotate(ciphertext, index);`
    
    This operation maintains SSA form in MLIR while enabling optimized C++ emission.
  }];
  
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWECiphertext:$ciphertext,
    Openfhe_EvalKey:$evalKey
  );
}

// In-place multiplication operation (self-assignment style)  
def MulInPlaceOp : Openfhe_Op<"mul_inplace"> {
  let summary = "In-place multiplication operation that modifies the first ciphertext";
  let description = [{
    Performs multiplication in-place, modifying the first input ciphertext. This operation
    is used to model self-assignment in C++ code generation:
    `ciphertext1 = cc->EvalMult(ciphertext1, ciphertext2);`
    
    This operation maintains SSA form in MLIR while enabling optimized C++ emission.
  }];
  
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWECiphertext:$lhs,
    NewLWECiphertext:$rhs
  );
}

// In-place multiplication with plaintext (self-assignment style)
def MulPlainInPlaceOp : Openfhe_Op<"mul_plain_inplace"> {
  let summary = "In-place multiplication of ciphertext and plaintext";
  let description = [{
    Performs multiplication with plaintext in-place, modifying the input ciphertext.
    This operation is used to model self-assignment in C++ code generation:
    `ciphertext = cc->EvalMult(ciphertext, plaintext);`
  }];
  
  let arguments = (ins
    Openfhe_CryptoContext:$cryptoContext,
    NewLWECiphertext:$ciphertext,
    NewLWEPlaintext:$plaintext
  );
}

#endif  // LIB_DIALECT_OPENFHE_IR_OPENFHEOPS_TD_
