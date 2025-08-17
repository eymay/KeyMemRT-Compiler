#ifndef LIB_TRANSFORMS_INCREMENTALMERGEEVALKEYS_INCREMENTALMERGEEVALKEYS_H_
#define LIB_TRANSFORMS_INCREMENTALMERGEEVALKEYS_INCREMENTALMERGEEVALKEYS_H_

#include "mlir/include/mlir/Pass/Pass.h" // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/IncrementalMergeEvalKeys/IncrementalMergeEvalKeys.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/IncrementalMergeEvalKeys/IncrementalMergeEvalKeys.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_INCREMENTALMERGEEVALKEYS_INCREMENTALMERGEEVALKEYS_H_