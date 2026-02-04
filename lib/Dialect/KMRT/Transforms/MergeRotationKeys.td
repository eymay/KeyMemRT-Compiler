#ifndef LIB_DIALECT_KMRT_TRANSFORMS_MERGEROTATIONKEYS_TD_
#define LIB_DIALECT_KMRT_TRANSFORMS_MERGEROTATIONKEYS_TD_

include "mlir/Pass/PassBase.td"

def MergeRotationKeys : Pass<"kmrt-merge-rotation-keys"> {
  let summary = "Merge rotation keys based on liveness analysis";
  let description = [{
    This pass optimizes rotation key management in KMRT by merging consecutive
    load_key and clear_key pairs for the same rotation index.

    Uses liveness analysis to determine when rotation keys can be reused instead
    of being cleared and reloaded. This reduces memory traffic and improves
    performance by keeping frequently used keys in memory longer.

    The pass works with both static and dynamic rotation keys:
    - Static keys: Identified by constant rotation indices in the type
    - Dynamic keys: Identified by analyzing affine loop induction variables

    Example transformation:

    Before:
    ```mlir
    %rk1 = kmrt.load_key %c5 : !kmrt.rot_key<rotation_index = 5>
    %ct1 = kmrt.rotation %ct0, %rk1 : ...
    kmrt.clear_key %rk1 : !kmrt.rot_key<rotation_index = 5>
    // ... few operations ...
    %rk2 = kmrt.load_key %c5 : !kmrt.rot_key<rotation_index = 5>
    %ct2 = kmrt.rotation %ct1, %rk2 : ...
    kmrt.clear_key %rk2 : !kmrt.rot_key<rotation_index = 5>
    ```

    After:
    ```mlir
    %rk1 = kmrt.load_key %c5 : !kmrt.rot_key<rotation_index = 5>
    %ct1 = kmrt.rotation %ct0, %rk1 : ...
    // Clear removed
    // ... few operations ...
    // Load removed - reusing %rk1
    %ct2 = kmrt.rotation %ct1, %rk1 : ...
    kmrt.clear_key %rk1 : !kmrt.rot_key<rotation_index = 5>
    ```

    The pass applies this optimization when the distance between clear and load
    operations is within a configurable threshold (default: 10 non-key-management
    operations).
  }];
  let dependentDialects = [
    "mlir::heir::kmrt::KMRTDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect"
  ];

  let options = [
    Option<"enableLoopPeeling", "enable-loop-peeling", "bool", "true",
           "Enable loop peeling optimization to extract key loads/clears">
  ];
}

#endif  // LIB_DIALECT_KMRT_TRANSFORMS_MERGEROTATIONKEYS_TD_
