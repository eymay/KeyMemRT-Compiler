#ifndef LIB_TRANSFORMS_SYMBOLICBSGSDECOMPOSITION_SYMBOLICBSGSDECOMPOSITION_TD_
#define LIB_TRANSFORMS_SYMBOLICBSGSDECOMPOSITION_SYMBOLICBSGSDECOMPOSITION_TD_

include "mlir/Pass/PassBase.td"

def SymbolicBSGSDecomposition : Pass<"symbolic-bsgs-decomposition", "func::FuncOp"> {
  let description = [{
      This pass identifies rotation operations inside affine loops where the
    rotation offset depends on the loop induction variable, and decomposes
    them using the Baby-Step Giant-Step (BSGS) algorithm.
    
    The decomposition is symbolic: N1 and N2 parameters are left as symbolic
    values annotated with metadata. A later optimization pass will determine
    the optimal concrete values for these parameters.

  }];
  let dependentDialects = [
    "mlir::heir::openfhe::OpenfheDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif  // LIB_TRANSFORMS_SYMBOLICBSGSDECOMPOSITION_SYMBOLICBSGSDECOMPOSITION_TD_
