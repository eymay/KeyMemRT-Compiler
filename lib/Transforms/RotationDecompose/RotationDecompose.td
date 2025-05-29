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
    
    This optimization can use two modes:
    1. Advanced optimization using a Mixed Integer Programming (MIP) approach through a Python
       script that finds the optimal base set that minimizes both the number of base 
       rotations and the depth of the computation graph for derived rotations. This mode
       respects user-specified constraints like base set size and chain length limits.
    2. Simple Baby-Step Giant-Step (BSGS) optimization that automatically determines the
       mathematically optimal BSGS parameters for the given rotation indices. This mode
       ignores user-specified size constraints and generates the theoretically optimal
       base set for BSGS, which works particularly well for contiguous ranges of indices.
  }];

  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];

  let options = [
    Option<"baseSetSize", "base-set-size", "unsigned", /*default=*/"10",
           "The size of the base rotation set to use (ignored in simple BSGS mode)">,
    Option<"maxChainLength", "max-chain-length", "unsigned", /*default=*/"5",
           "Maximum number of additions allowed in a chain (ignored in simple BSGS mode)">,
    Option<"pythonScript", "python-script", "std::string", /*default=*/"\"/home/eymen/Documents/keymemrt_project/heir/lib/Transforms/RotationDecompose/rotation_optimizer.py\"",
           "Path to the Python optimizer script">,
    Option<"timeLimit", "time-limit", "unsigned", /*default=*/"300",
           "Time limit for the optimizer in seconds (ignored in simple BSGS mode)">,
    Option<"useSimpleBsgs", "use-simple-bsgs", "bool", /*default=*/"false",
           "Use simple BSGS instead of the advanced MIP optimizer. This automatically determines optimal parameters and ignores base-set-size, max-chain-length, and min-range-length.">,
    Option<"minRangeLength", "min-range-length", "unsigned", /*default=*/"3",
           "Minimum length of contiguous range to apply BSGS (ignored in simple BSGS mode, only used with advanced optimization)">
  ];
}

#endif  // LIB_TRANSFORMS_ROTATIONDECOMPOSE_ROTATIONDECOMPOSE_TD_
