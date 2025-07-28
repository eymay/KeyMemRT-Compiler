#ifndef LIB_TRANSFORMS_KEYPREFETCHING_KEYPREFETCHING_TD_
#define LIB_TRANSFORMS_KEYPREFETCHING_KEYPREFETCHING_TD_

include "mlir/Pass/PassBase.td"

def KeyPrefetching : Pass<"key-prefetching"> {
  let summary = "Insert enqueue_key operations before deserialize_key operations";
  let description = [{
    This pass creates an enqueue_key operation for every deserialize_key operation.
    It walks backwards from each deserialize to find the optimal placement point
    based on cumulative operation costs. The enqueue is placed when the accumulated
    cost reaches the prefetch threshold.
    
    Uses the depth attribute from deserialize_key operations for cost scaling.
    Non-computational operations have 0 cost.
    
    The goal is to balance memory usage (not too early) with performance (not too late).
  }];

  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];

  let options = [
    Option<"prefetchThreshold", "prefetch-threshold", "unsigned", /*default=*/"50",
           "Cumulative cost threshold for enqueue placement">
  ];
}

#endif  // LIB_TRANSFORMS_KEYPREFETCHING_KEYPREFETCHING_TD_
