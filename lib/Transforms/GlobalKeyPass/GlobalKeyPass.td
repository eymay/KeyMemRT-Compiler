#ifndef LIB_TRANSFORMS_GLOBALKEYPASS_GLOBALKEYPASS_TD_
#define LIB_TRANSFORMS_GLOBALKEYPASS_GLOBALKEYPASS_TD_

include "mlir/Pass/PassBase.td"

def GlobalKeyPass : Pass<"convert-to-global-keys", "func::FuncOp"> {
  let summary = "Convert deserialize_key and rotate operations to global key versions";
  let description = [{
    This pass converts OpenFHE operations that use explicit evaluation keys as dataflow
    elements to global key versions that handle keys opaquely. This transformation makes
    outlining easier by removing key dependencies from the dataflow.

    The pass performs the following conversions:
    - openfhe.deserialize_key -> openfhe.deserialize_key_global  
    - openfhe.rot -> openfhe.rotate_global

    The global versions of these operations manage keys internally without exposing
    them as SSA values, simplifying the IR for outlining and code generation.

    Example transformation:
    ```mlir
    // Before:
    %key = openfhe.deserialize_key %cc, 1 : !openfhe.crypto_context -> !openfhe.eval_key
    %result = openfhe.rot %cc, %ct, %key : !openfhe.crypto_context, !openfhe.lwe_ciphertext, !openfhe.eval_key -> !openfhe.lwe_ciphertext

    // After:
    openfhe.deserialize_key_global %cc, 1 : !openfhe.crypto_context
    %result = openfhe.rotate_global %cc, %ct, 1 : !openfhe.crypto_context, !openfhe.lwe_ciphertext -> !openfhe.lwe_ciphertext
    ```
  }];

  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];
}

#endif  // LIB_TRANSFORMS_GLOBALKEYPASS_GLOBALKEYPASS_TD_
