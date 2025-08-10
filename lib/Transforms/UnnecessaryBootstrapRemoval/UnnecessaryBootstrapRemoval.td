#ifndef LIB_TRANSFORMS_UNNECESSARYBOOTSTRAPREMOVAL_UNNECESSARYBOOTSTRAPREMOVAL_TD_
#define LIB_TRANSFORMS_UNNECESSARYBOOTSTRAPREMOVAL_UNNECESSARYBOOTSTRAPREMOVAL_TD_

include "mlir/Pass/PassBase.td"

def UnnecessaryBootstrapRemoval : Pass<"remove-unnecessary-bootstraps"> {
  let summary = "Remove bootstrap operations that don't change tower count";
  let description = [{
    This pass removes unnecessary bootstrap operations that have the same
    number of towers in their input and output ciphertexts. 
    
    The pass relies on profile annotations from the ProfileAnnotator pass
    to determine the tower counts. It identifies bootstrap operations where:
    1. The input operand has X towers (from result_towers attribute on the defining op)
    2. The bootstrap operation has X towers (from result_towers attribute)
    3. X == X, making the bootstrap unnecessary for tower management
    
    Such bootstraps are redundant and can be removed as an optimization.
    The pass only operates on properly annotated IR and will skip bootstraps
    without complete tower information to preserve correctness.
  }];
  
  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];
}

#endif  // LIB_TRANSFORMS_UNNECESSARYBOOTSTRAPREMOVAL_UNNECESSARYBOOTSTRAPREMOVAL_TD_
