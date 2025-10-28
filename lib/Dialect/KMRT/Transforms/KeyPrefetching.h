#ifndef LIB_DIALECT_KMRT_TRANSFORMS_KEYPREFETCHING_H_
#define LIB_DIALECT_KMRT_TRANSFORMS_KEYPREFETCHING_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace heir {
namespace kmrt {

#define GEN_PASS_DECL_KMRTKEYPREFETCHING
#include "lib/Dialect/KMRT/Transforms/Passes.h.inc"

}  // namespace kmrt
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_KMRT_TRANSFORMS_KEYPREFETCHING_H_
