#ifndef LIB_DIALECT_KMRT_IR_KMRTDIALECT_TD_
#define LIB_DIALECT_KMRT_IR_KMRTDIALECT_TD_

include "mlir/IR/OpBase.td"

def KMRT_Dialect : Dialect {
  let name = "kmrt";
  let summary = "A dialect for KeyMemRT operations";
  let description = [{
    The KeyMemRT (KMRT) dialect defines types and operations for 
    key-memory runtime operations.
  }];

  let cppNamespace = "::mlir::heir::kmrt";
  let useDefaultTypePrinterParser = 1;
}

#endif // LIB_DIALECT_KMRT_IR_KMRTDIALECT_TD_
