#ifndef LIB_TRANSFORMS_LOOPTILING_LOOPTILING_TD_
#define LIB_TRANSFORMS_LOOPTILING_LOOPTILING_TD_

include "mlir/Pass/PassBase.td"

def LoopTiling : Pass<"loop-tiling", "func::FuncOp"> {
  let summary = "Tile BSGS loops to enable hoisting of loop-invariant operations";
  let description = [{
    This pass tiles affine loops containing BSGS-decomposed rotations by
    splitting them into outer and inner loops based on the BSGS tile size (N2).

    The tiling creates a structure where loop-invariant operations (like giant-step
    rotations that depend only on the outer loop variable) can be hoisted by
    subsequent LICM passes.

    Input: Single affine.for loop with BSGS decomposition pattern
    Output: Nested affine.for loops (outer and inner) with affine.if guards
  }];

  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif  // LIB_TRANSFORMS_LOOPTILING_LOOPTILING_TD_
