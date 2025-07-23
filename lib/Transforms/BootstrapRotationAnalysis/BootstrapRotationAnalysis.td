#ifndef LIB_TRANSFORMS_BOOTSTRAP_ROTATION_ANALYSIS_TD_
#define LIB_TRANSFORMS_BOOTSTRAP_ROTATION_ANALYSIS_TD_

include "mlir/Pass/PassBase.td"

def BootstrapRotationAnalysis : Pass<"bootstrap-rotation-analysis"> {
  let summary = "Analyze bootstrap operations and calculate required rotation indices";
  let description = [{
    This pass analyzes bootstrap operations in the OpenFHE dialect and calculates
    the required rotation indices based on the bootstrap setup parameters.
    
    The pass:
    1. Finds openfhe::SetupBootstrapOp to extract bootstrap parameters
    2. Calculates the rotation indices needed for bootstrap operations
    3. Adds these indices as attributes to openfhe::BootstrapOp operations
    
    The rotation indices are calculated using the same logic as OpenFHE's
    EvalBootstrapSetup, taking into account:
    - Number of slots
    - Cyclotomic order
    - Level budget for encoding and decoding
    - Dimension parameters (dim1)
    
    This information can be used by subsequent passes for:
    - Key management optimization
    - Rotation key generation
    - Memory usage analysis
    - Bootstrap performance optimization
  }];
  
  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];
}

#endif  // LIB_TRANSFORMS_BOOTSTRAP_ROTATION_ANALYSIS_TD_
