#ifndef LIB_TRANSFORMS_FHEFUNCTIONOUTLINING_FHEFUNCTIONOUTLINING_H_
#define LIB_TRANSFORMS_FHEFUNCTIONOUTLINING_FHEFUNCTIONOUTLINING_H_

#include <memory>
#include <vector>

#include "llvm/include/llvm/ADT/DenseMap.h"
#include "llvm/include/llvm/ADT/DenseSet.h"
#include "llvm/include/llvm/ADT/SmallVector.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Support/LLVM.h"

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/FHEFunctionOutlining/FHEFunctionOutlining.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/FHEFunctionOutlining/FHEFunctionOutlining.h.inc"

std::unique_ptr<Pass> createFHEFunctionOutliningPass();

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_FHEFUNCTIONOUTLINING_FHEFUNCTIONOUTLINING_H_
