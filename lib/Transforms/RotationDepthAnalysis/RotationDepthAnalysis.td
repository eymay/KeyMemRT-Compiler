#ifndef LIB_ANALYSIS_ROTATIONDEPTHANALYSIS_ROTATIONDEPTHANALYSIS_TD_
#define LIB_ANALYSIS_ROTATIONDEPTHANALYSIS_ROTATIONDEPTHANALYSIS_TD_

include "mlir/Pass/PassBase.td"

def RotationDepthAnalysis : Pass<"rotation-depth-analysis"> {
  let summary = "Analyze rotation operations at each multiplicative depth";
  let description = [{
    This pass analyzes programs to identify the multiplicative depth of each operation
    and counts the number of rotation operations occurring at each depth level.
    The pass prints a summary showing the distribution of rotations across depth levels,
    which is useful for performance analysis of homomorphic encryption operations.
  }];
  
  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];
}

#endif  // LIB_ANALYSIS_ROTATIONDEPTHANALYSIS_ROTATIONDEPTHANALYSIS_TD_
