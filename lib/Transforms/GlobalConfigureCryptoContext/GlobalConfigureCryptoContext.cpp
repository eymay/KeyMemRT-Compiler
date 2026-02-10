#include "lib/Transforms/GlobalConfigureCryptoContext/GlobalConfigureCryptoContext.h"

#include <algorithm>
#include <set>
#include <string>

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Support/LogicalResult.h"

namespace mlir {
namespace heir {

struct GlobalConfig {
  // Union of all features from input files
  bool hasRelinOp = false;
  bool hasBootstrapOp = false;
  bool supportFHE = false;  // OR of all supportFHE flags
  std::set<int64_t> rotIndices;

  // Max values for performance parameters
  int64_t maxMulDepth = 0;

  // Encryption parameters - should be consistent across files
  int64_t plaintextModulus = 0;

  // Safety margin for depth
  int64_t depthMargin = 5;

  // Bootstrap attributes to preserve
  int32_t cyclotomicOrder = 0;
  int32_t slots = 0;
  std::vector<int32_t> bootstrapRotationIndices;
  int64_t levelBudgetEncode = 3;
  int64_t levelBudgetDecode = 3;
};

#define GEN_PASS_DEF_GLOBALCONFIGURECRYPTOCONTEXT
#include "lib/Transforms/GlobalConfigureCryptoContext/GlobalConfigureCryptoContext.h.inc"

struct GlobalConfigureCryptoContext
    : impl::GlobalConfigureCryptoContextBase<GlobalConfigureCryptoContext> {
  using GlobalConfigureCryptoContextBase::GlobalConfigureCryptoContextBase;

  LogicalResult extractConfigFromGenerateFunction(func::FuncOp generateFunc,
                                                  GlobalConfig& globalConfig) {
    bool foundGenParamsOp = false;

    generateFunc.walk([&](Operation* op) {
      if (auto genParamsOp = dyn_cast<openfhe::GenParamsOp>(op)) {
        foundGenParamsOp = true;

        // Extract multiplicative depth and take maximum
        if (auto mulDepthAttr = genParamsOp.getMulDepthAttr()) {
          globalConfig.maxMulDepth =
              std::max(globalConfig.maxMulDepth, mulDepthAttr.getInt());
        }

        // Extract plaintext modulus (should be consistent)
        if (auto ptModAttr = genParamsOp.getPlainModAttr()) {
          int64_t newPtMod = ptModAttr.getInt();
          if (globalConfig.plaintextModulus == 0) {
            globalConfig.plaintextModulus = newPtMod;
          } else if (globalConfig.plaintextModulus != newPtMod) {
            op->emitWarning("Inconsistent plaintext modulus found: ")
                << newPtMod << " vs " << globalConfig.plaintextModulus;
          }
        }
      }

      // Check for GenContextOp to extract supportFHE flag
      if (auto genContextOp = dyn_cast<openfhe::GenContextOp>(op)) {
        if (auto supportFHEAttr = genContextOp.getSupportFHEAttr()) {
          globalConfig.supportFHE =
              globalConfig.supportFHE || supportFHEAttr.getValue();
        }
      }
    });

    return success(foundGenParamsOp);
  }

  // Extract configuration from configure_crypto_context functions
  LogicalResult extractConfigFromConfigureFunction(func::FuncOp configureFunc,
                                                   GlobalConfig& globalConfig) {
    configureFunc.walk([&](Operation* op) {
      // Check for GenMulKeyOp to detect relinearization needs
      if (auto genMulKeyOp = dyn_cast<openfhe::GenMulKeyOp>(op)) {
        globalConfig.hasRelinOp = true;
      }

      // Check for GenRotKeyOp and collect rotation indices
      if (auto genRotKeyOp = dyn_cast<openfhe::GenRotKeyOp>(op)) {
        if (auto indicesAttr = genRotKeyOp.getIndicesAttr()) {
          for (auto index : indicesAttr.asArrayRef()) {
            globalConfig.rotIndices.insert(index);
          }
        }
      }

      // Check for bootstrap operations
      if (isa<openfhe::BootstrapOp>(op)) {
        globalConfig.hasBootstrapOp = true;
      }

      // Check for SetupBootstrapOp and extract level budgets
      if (auto setupBootstrapOp = dyn_cast<openfhe::SetupBootstrapOp>(op)) {
        globalConfig.hasBootstrapOp = true;

        if (auto levelBudgetEncodeAttr =
                setupBootstrapOp.getLevelBudgetEncodeAttr()) {
          globalConfig.levelBudgetEncode = levelBudgetEncodeAttr.getInt();
        }
        if (auto levelBudgetDecodeAttr =
                setupBootstrapOp.getLevelBudgetDecodeAttr()) {
          globalConfig.levelBudgetDecode = levelBudgetDecodeAttr.getInt();
        }
      }

      // Check for GenBootstrapKeyOp and extract bootstrap attributes
      if (auto genBootstrapKeyOp = dyn_cast<openfhe::GenBootstrapKeyOp>(op)) {
        globalConfig.hasBootstrapOp = true;

        // Extract cyclotomic_order
        if (auto cyclotomicAttr = genBootstrapKeyOp->getAttrOfType<IntegerAttr>(
                "cyclotomic_order")) {
          globalConfig.cyclotomicOrder = cyclotomicAttr.getInt();
        }

        // Extract slots
        if (auto slotsAttr =
                genBootstrapKeyOp->getAttrOfType<IntegerAttr>("slots")) {
          globalConfig.slots = slotsAttr.getInt();
        }

        // Extract bootstrap rotation indices
        if (auto rotationIndicesAttr =
                genBootstrapKeyOp->getAttrOfType<ArrayAttr>(
                    "rotation_indices")) {
          for (auto attr : rotationIndicesAttr.getAsRange<IntegerAttr>()) {
            globalConfig.bootstrapRotationIndices.push_back(attr.getInt());
          }
        }
      }
    });

    return success();
  }

  // Create a new global generate function
  func::FuncOp createGlobalGenerateFunction(ModuleOp moduleOp,
                                            const GlobalConfig& config) {
    OpBuilder builder(moduleOp.getContext());
    builder.setInsertionPointToEnd(&moduleOp.getRegion().back());

    // Create function signature: () -> !cc
    auto cryptoContextType =
        openfhe::CryptoContextType::get(builder.getContext());
    auto functionType = builder.getFunctionType({}, {cryptoContextType});

    // Create the function
    auto globalGenerateFunc = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "alexnet_global__generate_crypto_context",
        functionType);

    // Build function body
    Block* entryBlock = globalGenerateFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    // Generate parameters with merged configuration (max depth + safety margin)
    int64_t finalMulDepth = config.maxMulDepth + config.depthMargin;

    auto genParamsOp = builder.create<openfhe::GenParamsOp>(
        builder.getUnknownLoc(),
        openfhe::CCParamsType::get(builder.getContext()),
        builder.getI64IntegerAttr(finalMulDepth),
        builder.getI64IntegerAttr(config.plaintextModulus));

    // Generate context with merged supportFHE flag
    auto genContextOp = builder.create<openfhe::GenContextOp>(
        builder.getUnknownLoc(), cryptoContextType, genParamsOp.getResult(),
        builder.getBoolAttr(config.supportFHE));

    // Return the context
    builder.create<func::ReturnOp>(builder.getUnknownLoc(),
                                   ValueRange{genContextOp.getResult()});

    return globalGenerateFunc;
  }

  // Create a new global configure function
  func::FuncOp createGlobalConfigureFunction(ModuleOp moduleOp,
                                             const GlobalConfig& config) {
    OpBuilder builder(moduleOp.getContext());
    builder.setInsertionPointToEnd(&moduleOp.getRegion().back());

    // Create function signature: (!cc, !sk) -> !cc
    auto cryptoContextType =
        openfhe::CryptoContextType::get(builder.getContext());
    auto privateKeyType = openfhe::PrivateKeyType::get(builder.getContext());
    auto functionType = builder.getFunctionType(
        {cryptoContextType, privateKeyType}, {cryptoContextType});

    // Create the function
    auto globalConfigFunc = builder.create<func::FuncOp>(
        builder.getUnknownLoc(), "alexnet_global__configure_crypto_context",
        functionType);

    // Build function body
    Block* entryBlock = globalConfigFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    Value cryptoContext = entryBlock->getArgument(0);
    Value privateKey = entryBlock->getArgument(1);
    Value updatedContext = cryptoContext;

    // Generate relinearization keys if needed
    if (config.hasRelinOp) {
      auto genMulKeyOp = builder.create<openfhe::GenMulKeyOp>(
          builder.getUnknownLoc(), updatedContext, privateKey);
    }

    // Generate rotation keys with all collected indices
    if (!config.rotIndices.empty()) {
      SmallVector<int64_t> indexVec(config.rotIndices.begin(),
                                    config.rotIndices.end());

      auto genRotKeyOp = builder.create<openfhe::GenRotKeyOp>(
          builder.getUnknownLoc(), updatedContext, privateKey,
          builder.getDenseI64ArrayAttr(indexVec));
      genRotKeyOp->setAttr("bootstrap_enabled",
                           builder.getBoolAttr(config.hasBootstrapOp));
    }

    // Generate bootstrap setup and keys if needed
    if (config.hasBootstrapOp) {
      // Add setup_bootstrap operation (using the extracted level budgets)
      auto setupBootstrapOp = builder.create<openfhe::SetupBootstrapOp>(
          builder.getUnknownLoc(), updatedContext,
          builder.getIntegerAttr(builder.getIndexType(),
                                 config.levelBudgetEncode),
          builder.getIntegerAttr(builder.getIndexType(),
                                 config.levelBudgetDecode));

      // Add gen_bootstrapkey operation with all preserved attributes
      auto genBootstrapKeyOp = builder.create<openfhe::GenBootstrapKeyOp>(
          builder.getUnknownLoc(), updatedContext, privateKey);

      // Add bootstrap attributes if they were extracted
      if (config.cyclotomicOrder > 0) {
        genBootstrapKeyOp->setAttr(
            "cyclotomic_order",
            builder.getI32IntegerAttr(config.cyclotomicOrder));
      }

      if (config.slots > 0) {
        genBootstrapKeyOp->setAttr("slots",
                                   builder.getI32IntegerAttr(config.slots));
      }

      if (!config.bootstrapRotationIndices.empty()) {
        SmallVector<Attribute> rotationAttrs;
        for (auto index : config.bootstrapRotationIndices) {
          rotationAttrs.push_back(builder.getI32IntegerAttr(index));
        }
        genBootstrapKeyOp->setAttr("rotation_indices",
                                   builder.getArrayAttr(rotationAttrs));
        genBootstrapKeyOp->setAttr(
            "num_rotation_indices",
            builder.getI32IntegerAttr(config.bootstrapRotationIndices.size()));
      }
    }

    // Return the configured context
    builder.create<func::ReturnOp>(builder.getUnknownLoc(),
                                   ValueRange{updatedContext});

    return globalConfigFunc;
  }

  // Remove individual functions to avoid conflicts
  void removeIndividualFunctions(ModuleOp moduleOp) {
    SmallVector<func::FuncOp> toErase;

    moduleOp.walk([&](func::FuncOp funcOp) {
      StringRef funcName = funcOp.getName();
      if ((funcName.contains("__generate_crypto_context") ||
           funcName.contains("__configure_crypto_context")) &&
          !funcName.starts_with("alexnet_global__")) {
        toErase.push_back(funcOp);
      }
    });

    for (auto func : toErase) {
      func.erase();
    }
  }

 public:
  void runOnOperation() override {
    auto moduleOp = cast<ModuleOp>(getOperation());
    GlobalConfig globalConfig;

    // Collect configurations from all generate functions
    WalkResult result = moduleOp.walk([&](func::FuncOp funcOp) {
      StringRef funcName = funcOp.getName();
      if (funcName.contains("__generate_crypto_context")) {
        if (failed(extractConfigFromGenerateFunction(funcOp, globalConfig))) {
          funcOp.emitError(
              "Failed to extract configuration from generate function: ")
              << funcName;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    // Collect configurations from all configure functions
    result = moduleOp.walk([&](func::FuncOp funcOp) {
      StringRef funcName = funcOp.getName();
      if (funcName.contains("__configure_crypto_context")) {
        if (failed(extractConfigFromConfigureFunction(funcOp, globalConfig))) {
          funcOp.emitError(
              "Failed to extract configuration from configure function: ")
              << funcName;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    // Create the global functions
    createGlobalGenerateFunction(moduleOp, globalConfig);
    createGlobalConfigureFunction(moduleOp, globalConfig);

    // Remove individual functions
    removeIndividualFunctions(moduleOp);

    // Log the merged configuration
    llvm::errs() << "Global crypto context configuration:\n";
    llvm::errs() << "  Max mul depth found: " << globalConfig.maxMulDepth
                 << "\n";
    llvm::errs() << "  Final mul depth (with margin): "
                 << (globalConfig.maxMulDepth + globalConfig.depthMargin)
                 << "\n";
    llvm::errs() << "  Support FHE: " << globalConfig.supportFHE << "\n";
    llvm::errs() << "  Has relinearization: " << globalConfig.hasRelinOp
                 << "\n";
    llvm::errs() << "  Has bootstrap: " << globalConfig.hasBootstrapOp << "\n";
    llvm::errs() << "  Rotation indices count: "
                 << globalConfig.rotIndices.size() << "\n";
    llvm::errs() << "  Plaintext modulus: " << globalConfig.plaintextModulus
                 << "\n";

    if (globalConfig.hasBootstrapOp) {
      llvm::errs() << "  Bootstrap configuration:\n";
      llvm::errs() << "    Cyclotomic order: " << globalConfig.cyclotomicOrder
                   << "\n";
      llvm::errs() << "    Slots: " << globalConfig.slots << "\n";
      llvm::errs() << "    Level budget encode: "
                   << globalConfig.levelBudgetEncode << "\n";
      llvm::errs() << "    Level budget decode: "
                   << globalConfig.levelBudgetDecode << "\n";
      llvm::errs() << "    Bootstrap rotation indices: "
                   << globalConfig.bootstrapRotationIndices.size() << "\n";
    }
  }
};

}  // namespace heir
}  // namespace mlir
