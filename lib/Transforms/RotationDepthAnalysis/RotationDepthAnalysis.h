#ifndef LIB_ANALYSIS_ROTATIONDEPTHANALYSIS_ROTATIONDEPTHANALYSIS_H_
#define LIB_ANALYSIS_ROTATIONDEPTHANALYSIS_ROTATIONDEPTHANALYSIS_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/RotationDepthAnalysis/RotationDepthAnalysis.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/RotationDepthAnalysis/RotationDepthAnalysis.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_ROTATIONDEPTHANALYSIS_ROTATIONDEPTHANALYSIS_H_
