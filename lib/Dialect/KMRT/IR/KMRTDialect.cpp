#include "lib/Dialect/KMRT/IR/KMRTDialect.h"

#include "lib/Dialect/KMRT/IR/KMRTOps.h"
#include "lib/Dialect/KMRT/IR/KMRTTypes.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"

#include "lib/Dialect/KMRT/IR/KMRTDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/KMRT/IR/KMRTTypes.cpp.inc"

namespace mlir {
namespace heir {
namespace kmrt {

void KMRTDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/KMRT/IR/KMRTTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/KMRT/IR/KMRTOps.cpp.inc"
      >();
}

} // namespace kmrt
} // namespace heir
} // namespace mlir
