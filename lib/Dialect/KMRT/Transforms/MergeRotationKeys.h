#ifndef LIB_DIALECT_KMRT_TRANSFORMS_MERGEROTATIONKEYS_H_
#define LIB_DIALECT_KMRT_TRANSFORMS_MERGEROTATIONKEYS_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace kmrt {

#define GEN_PASS_DECL_MERGEROTATIONKEYS
#include "lib/Dialect/KMRT/Transforms/Passes.h.inc"

}  // namespace kmrt
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_KMRT_TRANSFORMS_MERGEROTATIONKEYS_H_
