#ifndef LIB_TRANSFORMS_OPTIMIZE_ROTATION_TD_
#define LIB_TRANSFORMS_OPTIMIZE_ROTATION_TD_

include "mlir/Pass/PassBase.td"

def OptimizeRotation : Pass<"optimize-rotation"> {
  let summary = "Optimize rotation operations in FHE circuits";
  let description = [{
    This pass analyzes rotation operations in FHE circuits and optimizes them by finding
    a minimal set of base rotations. Instead of generating a rotation key for every 
    possible rotation index, it generates keys for a small "base set" of rotations,
    and implements other rotations as compositions of the base set.
    
    For example, instead of generating keys for rotations by 1, 2, 3, and 4 positions,
    we might generate keys only for rotations by 1 and 3 positions, and implement:
    - rotate(2) = rotate(1) + rotate(1)
    - rotate(4) = rotate(1) + rotate(3)
    
    This optimization uses an Integer Linear Programming (ILP) approach to find the
    optimal base set that minimizes both the number of base rotations and the depth
    of the computation graph for derived rotations.
  }];

  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];

  let options = [
    Option<"baseSetSize", "base-set-size", "int", /*default=*/"2",
           "The size of the base rotation set to use">,
    Option<"applyTransformation", "apply-transformation", "bool", /*default=*/"false",
           "Whether to actually transform the IR or just analyze it">
  ];
}

#endif  // LIB_TRANSFORMS_OPTIMIZE_ROTATION_TD_
