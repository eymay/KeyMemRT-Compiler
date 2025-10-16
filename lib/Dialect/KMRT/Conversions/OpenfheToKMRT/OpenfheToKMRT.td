#ifndef LIB_DIALECT_KMRT_CONVERSIONS_OPENFHETOKMRTT_OPENFHETOKMRTT_TD_
#define LIB_DIALECT_KMRT_CONVERSIONS_OPENFHETOKMRTT_OPENFHETOKMRTT_TD_

include "mlir/Pass/PassBase.td"

def OpenfheToKMRT : Pass<"openfhe-to-kmrt", "mlir::func::FuncOp"> {
  let summary = "Convert OpenFHE rotation operations to KMRT key management operations";
  let description = [{
    This pass converts OpenFHE rotation operations to KMRT operations with explicit
    key management. Each rotation operation is wrapped with load_key and clear_key
    operations to manage rotation keys explicitly.

    The conversion transforms:
    ```mlir
    %result = openfhe.rot %ciphertext, 5 : !openfhe.ciphertext -> !openfhe.ciphertext
    ```

    Into:
    ```mlir
    %rot_key = kmrt.load_key 5 : !kmrt.rot_key<5>
    %result = kmrt.rotation %ciphertext, %rot_key : !openfhe.ciphertext, !kmrt.rot_key<5> -> !openfhe.ciphertext
    kmrt.clear_key %rot_key : !kmrt.rot_key<5>
    ```
  }];

  let dependentDialects = [
    "mlir::heir::openfhe::OpenfheDialect",
    "mlir::keymemrt::kmrt::KMRTDialect"
  ];
}

#endif // LIB_DIALECT_KMRT_CONVERSIONS_OPENFHETOKMRTT_OPENFHETOKMRTT_TD_
