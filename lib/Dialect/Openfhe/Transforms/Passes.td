#ifndef LIB_DIALECT_OPENFHE_TRANSFORMS_PASSES_TD_
#define LIB_DIALECT_OPENFHE_TRANSFORMS_PASSES_TD_

include "mlir/Pass/PassBase.td"

def ConfigureCryptoContext : Pass<"openfhe-configure-crypto-context"> {
  let summary = "Configure the crypto context in OpenFHE";
  let description = [{
    This pass generates helper functions to generate and configure the OpenFHE crypto context for the given function. Generating the crypto context sets the appropriate encryption parameters, while the configuration generates the necessary evaluation keys (relinearization and rotation keys).

    For the options, reader can refer to the OpenFHE documentation at
    https://github.com/openfheorg/openfhe-development/blob/main/src/pke/examples/README.md#description-of-the-cryptocontext-parameters-and-their-restrictions

    For example, for an MLIR function `@my_func`, the generated helpers have the following signatures
    ```mlir
    func.func  @my_func__generate_crypto_context() -> !openfhe.crypto_context

    func.func  @my_func__configure_crypto_context(!openfhe.crypto_context, !openfhe.private_key) -> !openfhe.crypto_context
    ```
  }];
  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];
  let options = [
    Option<"entryFunction", "entry-function", "std::string",
           /*default=*/"", "Default entry function "
           "name of entry function.">,
    Option<"mulDepth", "mul-depth", "int",
           /*default=*/"0", "Manually specify the mul depth">,
    // Options for GenParamsOp
    Option<"ringDim", "ring-dim", "int",
           /*default=*/"0", "Manually specify the ring dimension (insecure is implied)">,
    // plaintext modulus is from IR
    Option<"batchSize", "batch-size", "int",
           /*default=*/"0", "Manually specify the batch size">,
    Option<"firstModSize", "first-mod-size", "int",
           /*default=*/"0", "Manually specify the first mod size">,
    Option<"scalingModSize", "scaling-mod-size", "int",
           /*default=*/"0", "Manually specify the scaling mod size">,
    Option<"digitSize", "digit-size", "int",
           /*default=*/"0", "Manually specify the digit size for relinearization">,
    Option<"numLargeDigits", "num-large-digits", "int",
           /*default=*/"0", "Manually specify the number of large digits for HYBRID relinearization">,
    Option<"maxRelinSkDeg", "max-relin-sk-deg", "int",
           /*default=*/"0", "Manually specify the max number of relin sk deg">,
    Option<"insecure", "insecure", "bool",
           /*default=*/"false", "Whether to use insecure parameter (defaults to false)">,
    Option<"keySwitchingTechniqueBV", "key-switching-technique-bv", "bool",
           /*default=*/"false", "Whether to use BV key switching technique (defaults to false)">,
    Option<"scalingTechniqueFixedManual", "scaling-technique-fixed-manual", "bool",
           /*default=*/"false", "Whether to use fixed manual scaling technique (defaults to false)">,
    // For bootstrapping
    Option<"levelBudgetEncode", "level-budget-encode", "int",
           /*default=*/"3", "Level budget for CKKS bootstrap encode (s2c) phase">,
    Option<"levelBudgetDecode", "level-budget-decode", "int",
           /*default=*/"3", "Level budget for CKKS bootstrap decode (c2s) phase">,
  ];
}

def CountAddAndKeySwitch : Pass<"openfhe-count-add-and-key-switch"> {
  let summary = "Count the number of add and key-switch operations in OpenFHE";
  let description = [{
    This pass counts the number of add and key-switch operations in the given function.

    This is used for setting the EvalAddCount and EvalKeySwitchCount in OpenFHE library.
    Cf. [Alexandru et al. 2024](https://eprint.iacr.org/2024/203) for why this
    is important for security.

    The detailed definition of these counts could be found in the KPZ21 paper
    [Revisiting Homomorphic Encryption Schemes for Finite Fields](https://ia.cr/2021/204)

    The pass should be run at the secret arithmetic level when management operations
    have been inserted and the IR is stable.
  }];
}

def ConvertToInPlace : Pass<"openfhe-convert-to-inplace"> {
  let summary = "Convert OpenFHE operations to their in-place equivalents";
  let description = [{
    This pass converts OpenFHE operations that return new values to their
    in-place equivalents that modify the first operand instead. This can
    help optimize memory usage by avoiding unnecessary copies.

    For example:
    ```mlir
    %result = openfhe.add %cc, %lhs, %rhs : (!openfhe.crypto_context, !lwe.lwe_ciphertext, !lwe.lwe_ciphertext) -> !lwe.lwe_ciphertext
    ```
    
    Becomes:
    ```mlir
    openfhe.add_inplace %cc, %lhs, %rhs : (!openfhe.crypto_context, !lwe.lwe_ciphertext, !lwe.lwe_ciphertext)
    // %lhs is now modified to contain %lhs + %rhs
    ```

    The pass only applies the transformation when it's safe to do so,
    typically when the result has only one use.
  }];
  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];
}

def InsertClearOps : Pass<"openfhe-insert-clear-ops"> {
  let summary = "Insert clear operations to reduce memory usage";
  let description = [{
    This pass analyzes the lifetime of ciphertext and plaintext values and inserts 
    clear operations at points where they are no longer needed. This helps reduce 
    memory usage in large FHE applications by explicitly releasing memory.

    The pass performs liveness analysis to determine when a ciphertext or plaintext
    value is last used and inserts clear operations immediately after that use:
    - clear_ct operations for ciphertexts
    - clear_pt operations for plaintexts

    This optimization is particularly beneficial for:
    - Large computation graphs with many intermediate ciphertexts and plaintexts
    - Functions with long sequences of operations where intermediate results
      can be freed early
    - Memory-constrained environments where FHE memory usage is a bottleneck

    The pass processes both ciphertext and plaintext values uniformly, providing
    comprehensive memory management for all FHE data types.
  }];
  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];
}

def FastRotationPrecompute : Pass<"openfhe-fast-rotation-precompute"> {
  let summary = "Identify and apply EvalFastRotation when possible.";
  let description = [{
    This pass identifies when a ciphertext is rotated by multiple different
    shifts, and replaces the `EvalRot` ops with `EvalFastRotationPrecompute`
    followed by `EvalFastRotate`.
  }];
}

#endif  // LIB_DIALECT_OPENFHE_TRANSFORMS_PASSES_TD_
