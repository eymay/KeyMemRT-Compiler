#ifndef LIB_TRANSFORMS_FHEFUNCTIONOUTLINING_FHEFUNCTIONOUTLINING_TD_
#define LIB_TRANSFORMS_FHEFUNCTIONOUTLINING_FHEFUNCTIONOUTLINING_TD_

include "mlir/Pass/PassBase.td"


def FHEFunctionOutlining : Pass<"fhe-function-outlining", "func::FuncOp"> {
  let summary = "Split FHE functions into smaller outlined functions";
  let description = [{
    This pass analyzes FHE functions at the CKKS level where linear_transform
    operations are still present, and splits them into smaller functions based
    on a specified split count parameter.

    The pass performs a simple DAG analysis to find good breakpoints that
    minimize dependencies between split segments while maintaining roughly
    equal node counts in each segment.

    For neural networks, this creates natural split points between layers
    or major computational blocks.

    The pass:
    1. Analyzes the function's operation DAG
    2. Finds optimal split points based on node count balance
    3. Creates separate outlined functions for each segment
    4. Generates a main dispatcher function that calls the segments in order
    5. Each outlined function operates on intermediate values and returns results

    Example transformation:
    ```mlir
    // Original large function
    func.func @original(%input: !ct) -> !ct {
      %layer1 = ckks.linear_transform %input, %weights1
      %act1 = ckks.mul %layer1, %layer1
      %layer2 = ckks.linear_transform %act1, %weights2
      %act2 = ckks.mul %layer2, %layer2
      %output = ckks.linear_transform %act2, %weights3
      return %output
    }

    // After outlining with split_count=3
    func.func @original_segment_0(%input: !ct) -> !ct {
      %layer1 = ckks.linear_transform %input, %weights1
      %act1 = ckks.mul %layer1, %layer1
      return %act1
    }

    func.func @original_segment_1(%input: !ct) -> !ct {
      %layer2 = ckks.linear_transform %input, %weights2
      %act2 = ckks.mul %layer2, %layer2
      return %act2
    }

    func.func @original_segment_2(%input: !ct) -> !ct {
      %output = ckks.linear_transform %input, %weights3
      return %output
    }

    func.func @original_main(%input: !ct) -> !ct {
      %seg0_result = call @original_segment_0(%input)
      %seg1_result = call @original_segment_1(%seg0_result)
      %seg2_result = call @original_segment_2(%seg1_result)
      return %seg2_result
    }
    ```
  }];

  let options = [
    Option<"splitCount", "split-count", "int",
           /*default=*/"2", "Number of segments to split the function into">,
  ];

  let dependentDialects = [
    "mlir::heir::ckks::CKKSDialect",
    "mlir::func::FuncDialect"
  ];
}
#endif  // LIB_TRANSFORMS_FHEFUNCTIONOUTLINING_FHEFUNCTIONOUTLINING_TD_
