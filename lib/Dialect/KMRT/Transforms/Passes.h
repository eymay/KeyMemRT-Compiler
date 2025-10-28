#ifndef LIB_DIALECT_KMRT_TRANSFORMS_PASSES_H_
#define LIB_DIALECT_KMRT_TRANSFORMS_PASSES_H_

#include "lib/Dialect/KMRT/IR/KMRTDialect.h"
#include "lib/Dialect/KMRT/Transforms/KeyPrefetching.h"
#include "lib/Dialect/KMRT/Transforms/MergeRotationKeys.h"

namespace mlir {
namespace heir {
namespace kmrt {

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/KMRT/Transforms/Passes.h.inc"

}  // namespace kmrt
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_KMRT_TRANSFORMS_PASSES_H_
