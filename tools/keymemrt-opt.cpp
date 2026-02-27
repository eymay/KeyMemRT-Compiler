#include <cstdlib>
#include <functional>
#include <memory>
#include <string>

// KeyMemRT dialects
#include "lib/Dialect/HEIRInterfaces.h"
#include "lib/Dialect/CKKS/Conversions/CKKSToLWE/CKKSToLWE.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/KMRT/IR/KMRTDialect.h"
#include "lib/Dialect/KMRT/Transforms/Passes.h"
#include "lib/Dialect/LWE/Conversions/LWEToOpenfhe/LWEToOpenfhe.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/Transforms/Passes.h"
#include "lib/Dialect/ModArith/IR/ModArithDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/Transforms/Passes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialDialect.h"
#include "lib/Dialect/RNS/IR/RNSDialect.h"
#include "lib/Dialect/RNS/IR/RNSTypeInterfaces.h"
#include "lib/Dialect/Random/IR/RandomDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"

// KeyMemRT transforms
#include "lib/Transforms/AnnotateModule/AnnotateModule.h"
#include "lib/Transforms/BootstrapRotationAnalysis/BootstrapRotationAnalysis.h"
#include "lib/Transforms/LowerLinearTransform/LowerLinearTransform.h"
#include "lib/Transforms/ProfileAnnotator/ProfileAnnotator.h"
#include "lib/Transforms/SymbolicBSGSDecomposition/SymbolicBSGSDecomposition.h"
#include "lib/Transforms/UnnecessaryBootstrapRemoval/UnnecessaryBootstrapRemoval.h"

// MLIR core
#include "mlir/include/mlir/Conversion/AffineToStandard/AffineToStandard.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Affine/Passes.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassManager.h"            // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"           // from @llvm-project

using namespace mlir;
using namespace heir;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register KeyMemRT dialects
  registry.insert<ckks::CKKSDialect>();
  registry.insert<kmrt::KMRTDialect>();
  registry.insert<lwe::LWEDialect>();
  registry.insert<mod_arith::ModArithDialect>();
  registry.insert<openfhe::OpenfheDialect>();
  registry.insert<::mlir::heir::polynomial::PolynomialDialect>();
  registry.insert<random::RandomDialect>();
  registry.insert<rns::RNSDialect>();
  registry.insert<tensor_ext::TensorExtDialect>();

  // Register MLIR dialects
  registry.insert<affine::AffineDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<scf::SCFDialect>();

  // Register MLIR passes
  registerTransformsPasses();      // canonicalize, cse, etc.
  affine::registerAffinePasses();  // loop unrolling, lower-affine

  // Register affine-to-standard conversion
  registerPass(
      []() -> std::unique_ptr<Pass> { return createLowerAffinePass(); });

  // Register KeyMemRT transforms
  kmrt::registerKMRTPasses();
  lwe::registerLWEPasses();
  openfhe::registerOpenfhePasses();
  registerAnnotateModulePasses();
  registerBootstrapRotationAnalysisPasses();
  registerLowerLinearTransformPasses();
  registerProfileAnnotatorPasses();
  registerSymbolicBSGSDecompositionPasses();
  registerUnnecessaryBootstrapRemovalPasses();

  // Register KeyMemRT conversions
  ckks::registerCKKSToLWEPasses();
  lwe::registerLWEToOpenfhePasses();

  // Register KeyMemRT interfaces
  rns::registerExternalRNSTypeInterfaces(registry);
  registerOperandAndResultAttrInterface(registry);

  return asMainReturnCode(
      MlirOptMain(argc, argv, "KeyMemRT Pass Driver", registry));
}
