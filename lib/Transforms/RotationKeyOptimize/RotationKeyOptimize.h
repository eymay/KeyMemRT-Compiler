#ifndef LIB_TRANSFORMS_ROTATION_KEY_OPTIMIZE_H_
#define LIB_TRANSFORMS_ROTATION_KEY_OPTIMIZE_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/RotationKeyOptimize/RotationKeyOptimize.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/RotationKeyOptimize/RotationKeyOptimize.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_OPTIMIZE_RELINEARIZATION_H_
