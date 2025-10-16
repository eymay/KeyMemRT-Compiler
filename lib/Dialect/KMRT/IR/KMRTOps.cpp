#include "lib/Dialect/KMRT/IR/KMRTOps.h"

#include "lib/Dialect/KMRT/IR/KMRTDialect.h"
#include "lib/Dialect/KMRT/IR/KMRTTypes.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/OpImplementation.h"
#include "llvm/include/llvm/ADT/SetVector.h"

#define GET_OP_CLASSES
#include "lib/Dialect/KMRT/IR/KMRTOps.cpp.inc"
