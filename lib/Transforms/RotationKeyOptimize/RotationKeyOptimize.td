#ifndef LIB_TRANSFORMS_ROTATION_KEY_OPTIMIZE_TD_
#define LIB_TRANSFORMS_ROTATION_KEY_OPTIMIZE_TD_

include "mlir/Pass/PassBase.td"

def RotationKeyOptimize : Pass<"rotation-key-optimize"> {
  let summary = "Optimize rotation key generation and usage based on liveness";
  let description = [{
    Uses liveness analysis to optimize when rotation keys are generated and freed.
    Currently just analyzes and dumps liveness information.
  }];
}
#endif  // LIB_TRANSFORMS_OPTIMIZE_RELINEARIZATION_TD_
