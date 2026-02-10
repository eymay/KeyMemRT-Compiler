#ifndef LIB_TRANSFORMS_MEMORYESTIMATION_MEMORYESTIMATION_H_
#define LIB_TRANSFORMS_MEMORYESTIMATION_MEMORYESTIMATION_H_

#include <map>
#include <string>

#include "llvm/include/llvm/ADT/DenseMap.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"     // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/MemoryEstimation/MemoryEstimation.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/MemoryEstimation/MemoryEstimation.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_MEMORYESTIMATION_MEMORYESTIMATION_H_
