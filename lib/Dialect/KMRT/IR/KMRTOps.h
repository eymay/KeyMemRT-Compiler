#ifndef LIB_DIALECT_KMRT_KMRTOPS_H_
#define LIB_DIALECT_KMRT_KMRTOPS_H_

#include "lib/Dialect/KMRT/IR/KMRTDialect.h"
#include "lib/Dialect/KMRT/IR/KMRTTypes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Dialect.h"

#define GET_OP_CLASSES
#include "lib/Dialect/KMRT/IR/KMRTOps.h.inc"

#endif // LIB_DIALECT_KMRT_KMRTOPS_H_
