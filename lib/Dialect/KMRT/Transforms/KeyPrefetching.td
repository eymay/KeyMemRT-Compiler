#ifndef LIB_DIALECT_KMRT_TRANSFORMS_KEYPREFETCHING_TD_
#define LIB_DIALECT_KMRT_TRANSFORMS_KEYPREFETCHING_TD_

include "mlir/Pass/PassBase.td"

def KMRTKeyPrefetching : Pass<"kmrt-key-prefetching"> {
  let summary = "Insert prefetch_key operations before load_key operations";
  let description = [{
    This pass inserts prefetch_key operations before load_key operations to
    enable asynchronous key loading and reduce latency.

    The pass has three modes:
    - For sequential code: Walk backwards from each load_key and insert prefetch_key
      when accumulated cost reaches the threshold
    - For affine loops: Create a separate prefetch loop before the main loop that
      prefetches all keys in the same order they are used in the main loop
    - Runtime delegated (--runtime-delegated): Collect all keys in execution order
      and emit them in a memref at the beginning of the program, allowing the runtime
      to fully control when keys are prefetched. The order is preserved exactly using
      affine loops and memref stores.

    For loops, the pass:
    1. Identifies all load_key operations in the loop
    2. Filters out keys that have a clear_key before the load_key (no need to prefetch)
    3. Creates a prefetch loop with the same bounds as the main loop
    4. In the prefetch loop, prefetches each key in the same order

    Operation costs are based on FHE operation complexity (for sequential code only):
    - Add/Sub: 1
    - Mul: 10
    - Rotation: 15
    - Bootstrap: 100
    - etc.

    The prefetch threshold (default: 50) determines the minimum cumulative cost
    of operations that should execute between prefetch and load in sequential code.

    Example transformation for sequential code:

    Before:
    ```mlir
    %v1 = openfhe.add %cc, %ct1, %ct2  // cost: 1
    %v2 = openfhe.mul %cc, %v1, %ct3   // cost: 10, cumulative: 11
    %v3 = openfhe.mul %cc, %v2, %ct4   // cost: 10, cumulative: 21
    %v4 = openfhe.mul %cc, %v3, %ct5   // cost: 10, cumulative: 31
    %v5 = openfhe.mul %cc, %v4, %ct6   // cost: 10, cumulative: 41
    %v6 = openfhe.mul %cc, %v5, %ct7   // cost: 10, cumulative: 51 >= 50
    %rk = kmrt.load_key %c5 : !kmrt.rot_key<rotation_index = 5>
    %ct_rot = openfhe.rot %cc, %v6, %rk
    kmrt.clear_key %rk
    ```

    After:
    ```mlir
    %c5 = arith.constant 5 : i64
    kmrt.prefetch_key %c5                 // Inserted before operations
    %v1 = openfhe.add %cc, %ct1, %ct2
    %v2 = openfhe.mul %cc, %v1, %ct3
    %v3 = openfhe.mul %cc, %v2, %ct4
    %v4 = openfhe.mul %cc, %v3, %ct5
    %v5 = openfhe.mul %cc, %v4, %ct6
    %v6 = openfhe.mul %cc, %v5, %ct7      // Prefetch placed here (cumulative cost ~50)
    %rk = kmrt.load_key %c5 : !kmrt.rot_key<rotation_index = 5>
    %ct_rot = openfhe.rot %cc, %v6, %rk
    kmrt.clear_key %rk
    ```

    Example transformation for affine loops:

    Before:
    ```mlir
    affine.for %iv = 1 to 10 {
      %iv_i64 = arith.index_cast %iv : index to i64
      %rk = kmrt.load_key %iv_i64 : !kmrt.rot_key
      %ct_rot = openfhe.rot %cc, %ct, %rk
      // ... more operations ...
    }
    ```

    After:
    ```mlir
    // Prefetch loop with same bounds
    affine.for %piv = 1 to 10 {
      %piv_i64 = arith.index_cast %piv : index to i64
      kmrt.prefetch_key %piv_i64
    }

    // Main loop (unchanged)
    affine.for %iv = 1 to 10 {
      %iv_i64 = arith.index_cast %iv : index to i64
      %rk = kmrt.load_key %iv_i64 : !kmrt.rot_key
      %ct_rot = openfhe.rot %cc, %ct, %rk
      // ... more operations ...
    }
    ```

    Keys that are cleared before being loaded are not prefetched:
    ```mlir
    // Before
    affine.for %iv = 1 to 10 {
      %iv_i64 = arith.index_cast %iv : index to i64
      kmrt.clear_key %iv_i64
      %rk = kmrt.load_key %iv_i64 : !kmrt.rot_key
      %ct_rot = openfhe.rot %cc, %ct, %rk
    }

    // After: No prefetch loop created (key is cleared before load)
    ```
  }];

  let dependentDialects = [
    "mlir::heir::kmrt::KMRTDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect",
  ];

  let options = [
    Option<"prefetchThreshold", "prefetch-threshold", "unsigned", /*default=*/"50",
           "Cumulative cost threshold for prefetch placement">,
    Option<"skipInsertion", "skip-insertion", "bool", /*default=*/"false",
           "Skip automatic prefetch insertion, only run verification (requires verify-prefetches=true)">,
    Option<"verifyPrefetches", "verify-prefetches", "bool", /*default=*/"false",
           "Verify that every load_key has a corresponding prefetch_key using abstract interpretation">,
    Option<"runtimeDelegated", "runtime-delegated", "bool", /*default=*/"false",
           "Collect all keys in execution order and delegate prefetching decisions to runtime">
  ];
}

#endif  // LIB_DIALECT_KMRT_TRANSFORMS_KEYPREFETCHING_TD_
