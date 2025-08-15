#ifndef LIB_TRANSFORMS_FUNCTIONEXTRACTOR_FUNCTIONEXTRACTOR_TD_
#define LIB_TRANSFORMS_FUNCTIONEXTRACTOR_FUNCTIONEXTRACTOR_TD_

include "mlir/Pass/PassBase.td"

def FunctionExtractor : Pass<"extract-function", "mlir::ModuleOp"> {
  let summary = "Extract a specific function and print it to stdout";
  let description = [{
    This pass extracts a specific function by name and prints only that function
    to stdout. This is useful for isolating functions after transformation passes
    like FHE function outlining, allowing you to work with individual segments.

    The pass:
    1. Searches for the specified function by name
    2. Creates a new module containing only that function
    3. Prints the module to stdout (standard MLIR pass behavior)

    Example usage:
    ```bash
    heir-opt --extract-function=function-name=myfunction input.mlir > myfunction.mlir
    heir-opt --extract-function=function-name=alexnet_computation_segment_0 outlined.mlir > segment0.mlir
    ```

    This follows the standard MLIR pass pattern where output goes to stdout
    and can be redirected as needed.
  }];

  let options = [
    Option<"functionName", "function-name", "std::string",
           /*default=*/"\"\"", "Name of the function to extract">
  ];

  let dependentDialects = [
    "mlir::func::FuncDialect"
  ];
}

#endif  // LIB_TRANSFORMS_FUNCTIONEXTRACTOR_FUNCTIONEXTRACTOR_TD_
