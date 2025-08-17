#ifndef LIB_TRANSFORMS_PATTERNLOOPROLLING_PATTERNLOOPROLLING_TD_
#define LIB_TRANSFORMS_PATTERNLOOPROLLING_PATTERNLOOPROLLING_TD_

include "mlir/Pass/PassBase.td"

def PatternLoopRolling : Pass<"pattern-loop-rolling", "func::FuncOp"> {
  // FIXME: add add summary/description
  let summary = "";
  let description = [{
  }];
  let dependentDialects = [

    "mlir::scf::SCFDialect",
  ];
}

#endif  // LIB_TRANSFORMS_PATTERNLOOPROLLING_PATTERNLOOPROLLING_TD_
