#include "lib/Target/Metadata/MetadataEmitter.h"
#include "lib/Target/OpenFhePke/OpenFheTranslateRegistration.h"
#include "llvm/include/llvm/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Tools/mlir-translate/MlirTranslateMain.h"  // from @llvm-project

int main(int argc, char **argv) {
  // Metadata output
  mlir::heir::registerMetadataEmitter();

  // OpenFHE output
  mlir::heir::openfhe::registerTranslateOptions();
  mlir::heir::openfhe::registerToOpenFhePkeTranslation();
  mlir::heir::openfhe::registerToOpenFhePkeHeaderTranslation();
  mlir::heir::openfhe::registerToOpenFhePkePybindTranslation();

  return failed(mlir::mlirTranslateMain(argc, argv, "KeyMemRT Translation Tool"));
}
