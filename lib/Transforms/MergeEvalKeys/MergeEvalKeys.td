#ifndef LIB_TRANSFORMS_MERGE_EVAL_KEYS_TD_
#define LIB_TRANSFORMS_MERGE_EVAL_KEYS_TD_

include "mlir/Pass/PassBase.td"

def MergeEvalKeys : Pass<"merge-eval-keys"> {
  let summary = "Merge evaluation keys based on liveness";
  let description = [{
    Uses liveness analysis to optimize when rotation keys are generated and freed.
    Merges consecutive deserialize and clear pairs.
  }];
}
#endif  // LIB_TRANSFORMS_OPTIMIZE_RELINEARIZATION_TD_
