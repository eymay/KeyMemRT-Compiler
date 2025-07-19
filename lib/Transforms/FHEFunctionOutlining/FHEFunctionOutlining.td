#ifndef LIB_TRANSFORMS_FHEFUNCTIONOUTLINING_FHEFUNCTIONOUTLINING_TD_
#define LIB_TRANSFORMS_FHEFUNCTIONOUTLINING_FHEFUNCTIONOUTLINING_TD_

include "mlir/Pass/PassBase.td"


def FHEFunctionOutlining : Pass<"fhe-function-outlining", "mlir::ModuleOp"> {
  let summary = "Outline FHE computations into separate functions for faster compilation";
  let description = [{
    This pass analyzes dataflow convergence patterns in large FHE functions and 
    extracts self-contained computational regions into separate functions. This 
    enables faster compilation by:
    
    1. Splitting large inlined functions into smaller, manageable pieces
    2. Allowing parallel compilation of extracted functions
    3. Enabling higher optimization levels on smaller functions
    4. Distributing functions across multiple MLIR files for modular compilation
    
    The pass uses sparse dataflow analysis to identify convergence points where
    multiple computation paths merge, which make good candidates for function
    boundaries. Each extracted region is validated to ensure it's self-contained
    and maintains program semantics.
    
    Example transformation:
    ```mlir
    func.func @large_fhe_function(%arg0: !lwe.lwe_ciphertext<...>) -> !lwe.lwe_ciphertext<...> {
      %0 = openfhe.mul %cc, %arg0, %arg0 : ... -> ...
      %1 = openfhe.add %cc, %0, %arg0 : ... -> ...
      // ... many more operations ...
      %100 = openfhe.rotate %cc, %99, 5 : ... -> ...
      return %100 : !lwe.lwe_ciphertext<...>
    }
    ```
    
    Becomes:
    ```mlir
    func.func @outlined_fhe_region_0_25ops(%arg0: !lwe.lwe_ciphertext<...>) -> !lwe.lwe_ciphertext<...> {
      %0 = openfhe.mul %cc, %arg0, %arg0 : ... -> ...
      %1 = openfhe.add %cc, %0, %arg0 : ... -> ...
      // ... first 25 operations ...
      return %24 : !lwe.lwe_ciphertext<...>
    }
    
    func.func @large_fhe_function(%arg0: !lwe.lwe_ciphertext<...>) -> !lwe.lwe_ciphertext<...> {
      %0 = call @outlined_fhe_region_0_25ops(%arg0) : ... -> ...
      %1 = call @outlined_fhe_region_1_30ops(%0) : ... -> ...
      // ... calls to other outlined functions ...
      return %final : !lwe.lwe_ciphertext<...>
    }
    ```
  }];
  
  let dependentDialects = [
    "mlir::func::FuncDialect"
  ];
}
#endif  // LIB_TRANSFORMS_FHEFUNCTIONOUTLINING_FHEFUNCTIONOUTLINING_TD_
