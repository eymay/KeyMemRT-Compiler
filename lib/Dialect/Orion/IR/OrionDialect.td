#ifndef LIB_DIALECT_ORION_IR_ORIONDIALECT_TD_
#define LIB_DIALECT_ORION_IR_ORIONDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Orion_Dialect : Dialect {
  let name = "orion";
  let description = [{
    The `orion` dialect is an entry dialect for the Orion compiler into the
    HEIR ecosystem. It mirrors upstream HEIR's `orion` dialect and hosts the
    ops the Orion frontend emits as black boxes (linear transforms and
    Chebyshev evaluations) that the CKKS dialect does not natively model.
  }];

  let cppNamespace = "::mlir::heir::orion";
}

#endif  // LIB_DIALECT_ORION_IR_ORIONDIALECT_TD_
