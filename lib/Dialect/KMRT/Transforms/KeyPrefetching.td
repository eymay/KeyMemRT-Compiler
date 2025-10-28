#ifndef LIB_DIALECT_KMRT_TRANSFORMS_KEYPREFETCHING_TD_
#define LIB_DIALECT_KMRT_TRANSFORMS_KEYPREFETCHING_TD_

include "mlir/Pass/PassBase.td"

def KMRTKeyPrefetching : Pass<"kmrt-key-prefetching"> {
  let summary = "Insert prefetch_key operations before load_key operations";
  let description = [{
    This pass inserts prefetch_key operations before load_key operations to
    enable asynchronous key loading and reduce latency.

    The pass analyzes operation costs and inserts prefetches at optimal points:
    - For sequential code: Walk backwards from each load_key and insert prefetch_key
      when accumulated cost reaches the threshold
    - For affine loops: Insert prefetches for the first N iterations before the loop,
      and inside the loop insert prefetches for iteration (i + N) wrapped in affine.if
      to avoid prefetching beyond loop bounds

    Operation costs are based on FHE operation complexity:
    - Add/Sub: 1
    - Mul: 10
    - Rotation: 15
    - Bootstrap: 100
    - etc.

    The prefetch distance is determined by the cost threshold (default: 50),
    which represents the minimum cumulative cost of operations that should
    execute between prefetch and load.

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
      kmrt.clear_key %rk
      // ... more operations ...
    }
    ```

    After (with prefetch distance of 3 iterations):
    ```mlir
    %c1 = arith.constant 1 : i64
    kmrt.prefetch_key %c1
    %c2 = arith.constant 2 : i64
    kmrt.prefetch_key %c2
    %c3 = arith.constant 3 : i64
    kmrt.prefetch_key %c3

    affine.for %iv = 1 to 10 {
      // Prefetch for iteration (i+3) if i+3 < 10
      affine.if affine_set<(d0) : (d0 + 3 < 10)>(%iv) {
        %next_iv = affine.apply affine_map<(d0) -> (d0 + 3)>(%iv)
        %next_iv_i64 = arith.index_cast %next_iv : index to i64
        kmrt.prefetch_key %next_iv_i64
      }

      %iv_i64 = arith.index_cast %iv : index to i64
      %rk = kmrt.load_key %iv_i64 : !kmrt.rot_key
      %ct_rot = openfhe.rot %cc, %ct, %rk
      kmrt.clear_key %rk
      // ... more operations ...
    }
    ```
  }];

  let dependentDialects = [
    "mlir::heir::kmrt::KMRTDialect",
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect"
  ];

  let options = [
    Option<"prefetchThreshold", "prefetch-threshold", "unsigned", /*default=*/"50",
           "Cumulative cost threshold for prefetch placement">,
    Option<"verifyPrefetches", "verify-prefetches", "bool", /*default=*/"false",
           "Verify that every load_key has a corresponding prefetch_key using abstract interpretation">
  ];
}

#endif  // LIB_DIALECT_KMRT_TRANSFORMS_KEYPREFETCHING_TD_
