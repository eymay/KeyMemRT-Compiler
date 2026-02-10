#ifndef LIB_TRANSFORMS_LLTTYPEINFERENCE_LLTTYPEINFERENCE_TD_
#define LIB_TRANSFORMS_LLTTYPEINFERENCE_LLTTYPEINFERENCE_TD_

include "mlir/Pass/PassBase.td"

def LLTTypeInference : Pass<"llt-type-inference", "mlir::ModuleOp"> {
  let summary = "Simulate linear transform lowering for correct type inference";
  let description = [{
    This pass simulates the lowering of linear_transform operations to 
    infer the correct scaling factors for the entire computation chain.
  }];
  
  let dependentDialects = [
    "mlir::heir::ckks::CKKSDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::func::FuncDialect"
  ];
}
#endif  // LIB_TRANSFORMS_LLTTYPEINFERENCE_LLTTYPEINFERENCE_TD_
