#ifndef LIB_TRANSFORMS_ROTATIONDECOMPOSE_ROTATIONDECOMPOSE_TD_
#define LIB_TRANSFORMS_ROTATIONDECOMPOSE_ROTATIONDECOMPOSE_TD_

include "mlir/Pass/PassBase.td"

def RotationDecompose : Pass<"rotation-decompose"> {
  let summary = "Optimize rotation operations by finding minimal base set";
  let description = [{
    This pass analyzes rotation operations in FHE circuits and optimizes them by finding
    a minimal set of base rotations. Instead of generating a rotation key for every 
    possible rotation index, it generates keys for a small "base set" of rotations,
    and implements other rotations as compositions of the base set.
    
    For example, instead of generating keys for rotations by 1, 2, 3, and 4 positions,
    we might generate keys only for rotations by 1 and 4 positions, and implement:
    - rotate(2) = rotate(1) + rotate(1)
    - rotate(3) = rotate(1) + rotate(1) + rotate(1)
    
    This optimization uses a Mixed Integer Programming (MIP) approach through a Python
    script that finds the optimal base set that minimizes both the number of base 
    rotations and the depth of the computation graph for derived rotations.
  }];

  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];

  let options = [
    Option<"baseSetSize", "base-set-size", "unsigned", /*default=*/"2",
           "The size of the base rotation set to use">,
    Option<"maxChainLength", "max-chain-length", "unsigned", /*default=*/"5",
           "Maximum number of additions allowed in a chain">,
    Option<"pythonScript", "python-script", "std::string", /*default=*/"\"lib/Transforms/RotationDecompose/rotation_optimizer.py\"",
           "Path to the Python optimizer script">,
    Option<"timeLimit", "time-limit", "unsigned", /*default=*/"300",
           "Time limit for the optimizer in seconds">
  ];
}

#endif  // LIB_TRANSFORMS_ROTATIONDECOMPOSE_ROTATIONDECOMPOSE_TD_
