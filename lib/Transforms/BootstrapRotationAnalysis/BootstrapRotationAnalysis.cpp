#include "lib/Transforms/BootstrapRotationAnalysis/BootstrapRotationAnalysis.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "lib/Dialect/KMRT/IR/KMRTOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "llvm/include/llvm/Support/Debug.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Support/LLVM.h"

namespace mlir {
namespace heir {

class BootstrapKeyConfigRegistry {
 public:
  using KeyLevelMap = std::map<int32_t, int>;
  using ConfigKey =
      std::pair<uint32_t, uint32_t>;  // {encode_budget, decode_budget}

 private:
  static std::map<ConfigKey, KeyLevelMap> configurations;

 public:
  // Register a new configuration
  static void registerConfiguration(uint32_t encodeBudget,
                                    uint32_t decodeBudget,
                                    const KeyLevelMap& keyLevels) {
    ConfigKey key = {encodeBudget, decodeBudget};
    configurations[key] = keyLevels;
    llvm::dbgs() << "Registered bootstrap key configuration {" << encodeBudget
                 << "," << decodeBudget << "} with " << keyLevels.size()
                 << " key mappings\n";
  }

  // Check if a configuration exists
  static bool hasConfiguration(uint32_t encodeBudget, uint32_t decodeBudget) {
    ConfigKey key = {encodeBudget, decodeBudget};
    return configurations.find(key) != configurations.end();
  }

  // Get level for a rotation index in a specific configuration
  static int getLevel(uint32_t encodeBudget, uint32_t decodeBudget,
                      int32_t rotationIndex) {
    ConfigKey key = {encodeBudget, decodeBudget};
    auto configIt = configurations.find(key);
    if (configIt == configurations.end()) {
      return -1;  // Configuration not found
    }

    auto levelIt = configIt->second.find(rotationIndex);
    return (levelIt != configIt->second.end()) ? levelIt->second : -1;
  }

  // Check if a rotation index has a predefined level in a configuration
  static bool hasLevel(uint32_t encodeBudget, uint32_t decodeBudget,
                       int32_t rotationIndex) {
    return getLevel(encodeBudget, decodeBudget, rotationIndex) != -1;
  }

  // Get all rotation indices for a configuration
  static std::vector<int32_t> getRotationIndices(uint32_t encodeBudget,
                                                 uint32_t decodeBudget) {
    ConfigKey key = {encodeBudget, decodeBudget};
    auto configIt = configurations.find(key);
    if (configIt == configurations.end()) {
      return {};
    }

    std::vector<int32_t> indices;
    for (const auto& entry : configIt->second) {
      indices.push_back(entry.first);
    }
    return indices;
  }

  // Get level distribution for a configuration
  static std::map<int, int> getLevelDistribution(uint32_t encodeBudget,
                                                 uint32_t decodeBudget) {
    ConfigKey key = {encodeBudget, decodeBudget};
    auto configIt = configurations.find(key);
    if (configIt == configurations.end()) {
      return {};
    }

    std::map<int, int> distribution;
    for (const auto& entry : configIt->second) {
      distribution[entry.second]++;
    }
    return distribution;
  }

  // Get configuration key as string for logging
  static std::string getConfigString(uint32_t encodeBudget,
                                     uint32_t decodeBudget) {
    return "{" + std::to_string(encodeBudget) + "," +
           std::to_string(decodeBudget) + "}";
  }
};

std::map<BootstrapKeyConfigRegistry::ConfigKey,
         BootstrapKeyConfigRegistry::KeyLevelMap>
    BootstrapKeyConfigRegistry::configurations;

// Configuration initializer - this is where you add new configurations
class BootstrapKeyConfigInitializer {
 public:
  static void initializeConfigurations() {
    // {3,3} Level Budget Configuration
    // Based on analysis showing bootstrap levels start from same low levels
    // due to modulus extension behavior
    BootstrapKeyConfigRegistry::KeyLevelMap config_3_3 = {
        // Level 2 keys
        {16384, 2},
        {2048, 2},
        {3072, 2},
        {4096, 2},
        {5120, 2},
        {6144, 2},
        {7168, 2},
        {8192, 2},
        {9216, 2},
        {10240, 2},
        {11264, 2},
        {12288, 2},
        {13312, 2},
        {14336, 2},
        {15360, 2},

        // Level 3 keys (higher resolution rotations)
        {512, 3},
        {1024, 3},
        {1536, 3},
        {31776, 3},
        {31808, 3},
        {31840, 3},
        {31872, 3},
        {31904, 3},
        {31936, 3},
        {31968, 3},
        {32000, 3},
        {32032, 3},
        {32064, 3},
        {32096, 3},
        {32128, 3},
        {32160, 3},
        {32192, 3},
        {32224, 3},
        {32256, 3},

        // Level 4 keys (finest granularity rotations)
        {16, 4},
        {32, 4},
        {48, 4},
        {32737, 4},
        {32738, 4},
        {32739, 4},
        {32740, 4},
        {32741, 4},
        {32742, 4},
        {32743, 4},
        {32744, 4},
        {32745, 4},
        {32746, 4},
        {32747, 4},
        {32748, 4},
        {32749, 4},
        {32750, 4},
        {32751, 4},
        {32752, 4}};

    BootstrapKeyConfigRegistry::registerConfiguration(3, 3, config_3_3);
  }
};

// Bootstrap utilities (from the provided code)
struct BootstrapParams {
  int32_t levelBudget;
  int32_t layersCollapse;
  int32_t remCollapse;
  int32_t numRotations;
  int32_t babyStep;
  int32_t giantStep;
  int32_t numRotationsRem;
  int32_t babyStepRem;
  int32_t giantStepRem;
};

std::vector<uint32_t> SelectLayers(uint32_t logSlots, uint32_t levelBudget) {
  if (levelBudget >= logSlots) {
    return {1, logSlots, 0};
  }
  uint32_t layersPerLevel = logSlots / levelBudget;
  uint32_t remainder = logSlots % levelBudget;
  return {layersPerLevel, levelBudget, remainder};
}

BootstrapParams GetCollapsedFFTParams(uint32_t slots, uint32_t levelBudget,
                                      uint32_t dim1) {
  uint32_t logSlots = static_cast<uint32_t>(std::log2(slots));
  if (logSlots == 0) {
    logSlots = 1;
  }

  std::vector<uint32_t> dims = SelectLayers(logSlots, levelBudget);

  int32_t layersCollapse = dims[0];
  int32_t remCollapse = dims[2];
  bool flagRem = (remCollapse == 0) ? false : true;

  uint32_t numRotations = (1 << (layersCollapse + 1)) - 1;
  uint32_t numRotationsRem = (1 << (remCollapse + 1)) - 1;

  int32_t g;
  if (dim1 == 0 || dim1 > numRotations) {
    if (numRotations > 7) {
      g = (1 << (static_cast<int32_t>(layersCollapse / 2) + 2));
    } else {
      g = (1 << (static_cast<int32_t>(layersCollapse / 2) + 1));
    }
  } else {
    g = dim1;
  }

  int32_t b = (numRotations + 1) / g;
  int32_t bRem = 0;
  int32_t gRem = 0;

  if (flagRem) {
    if (numRotationsRem > 7) {
      gRem = (1 << (static_cast<int32_t>(remCollapse / 2) + 2));
    } else {
      gRem = (1 << (static_cast<int32_t>(remCollapse / 2) + 1));
    }
    bRem = (numRotationsRem + 1) / gRem;
  }

  return {.levelBudget = static_cast<int32_t>(levelBudget),
          .layersCollapse = layersCollapse,
          .remCollapse = remCollapse,
          .numRotations = static_cast<int32_t>(numRotations),
          .babyStep = b,
          .giantStep = g,
          .numRotationsRem = static_cast<int32_t>(numRotationsRem),
          .babyStepRem = bRem,
          .giantStepRem = gRem};
}

int32_t ReduceRotation(int32_t rotation, uint32_t slots) {
  rotation = rotation % static_cast<int32_t>(slots);
  if (rotation < 0) {
    rotation += slots;
  }
  return rotation;
}

std::vector<int32_t> FindCoeffsToSlotsRotationIndices(
    uint32_t slots, uint32_t M, const BootstrapParams& params) {
  std::vector<int32_t> indexList;

  int32_t levelBudget = params.levelBudget;
  int32_t layersCollapse = params.layersCollapse;
  int32_t remCollapse = params.remCollapse;
  int32_t numRotations = params.numRotations;
  int32_t b = params.babyStep;
  int32_t g = params.giantStep;
  int32_t numRotationsRem = params.numRotationsRem;
  int32_t bRem = params.babyStepRem;
  int32_t gRem = params.giantStepRem;

  int32_t stop;
  int32_t flagRem;
  if (remCollapse == 0) {
    stop = -1;
    flagRem = 0;
  } else {
    stop = 0;
    flagRem = 1;
  }

  indexList.reserve(b + g - 2 + bRem + gRem - 2 + 1 + M);

  for (int32_t s = levelBudget - 1; s > stop; s--) {
    for (int32_t j = 0; j < g; j++) {
      indexList.emplace_back(ReduceRotation(
          (j - int32_t((numRotations + 1) / 2) + 1) *
              (1 << ((s - flagRem) * layersCollapse + remCollapse)),
          slots));
    }

    for (int32_t i = 0; i < b; i++) {
      indexList.emplace_back(ReduceRotation(
          (g * i) * (1 << ((s - flagRem) * layersCollapse + remCollapse)),
          M / 4));
    }
  }

  if (flagRem) {
    for (int32_t j = 0; j < gRem; j++) {
      indexList.emplace_back(
          ReduceRotation((j - int32_t((numRotationsRem + 1) / 2) + 1), slots));
    }
    for (int32_t i = 0; i < bRem; i++) {
      indexList.emplace_back(ReduceRotation(gRem * i, M / 4));
    }
  }

  uint32_t m = slots * 4;
  if (m != M) {
    for (uint32_t j = 1; j < M / m; j <<= 1) {
      indexList.emplace_back(j * slots);
    }
  }

  return indexList;
}

std::vector<int32_t> FindSlotsToCoeffsRotationIndices(
    uint32_t slots, uint32_t M, const BootstrapParams& params) {
  std::vector<int32_t> indexList;

  int32_t levelBudget = params.levelBudget;
  int32_t layersCollapse = params.layersCollapse;
  int32_t remCollapse = params.remCollapse;
  int32_t numRotations = params.numRotations;
  int32_t b = params.babyStep;
  int32_t g = params.giantStep;
  int32_t numRotationsRem = params.numRotationsRem;
  int32_t bRem = params.babyStepRem;
  int32_t gRem = params.giantStepRem;

  int32_t flagRem = (remCollapse == 0) ? 0 : 1;

  indexList.reserve(b + g - 2 + bRem + gRem - 2 + 1 + M);

  for (int32_t s = 0; s < levelBudget; s++) {
    for (int32_t j = 0; j < g; j++) {
      indexList.emplace_back(ReduceRotation(
          (j - (numRotations + 1) / 2 + 1) * (1 << (s * layersCollapse)),
          M / 4));
    }
    for (int32_t i = 0; i < b; i++) {
      indexList.emplace_back(
          ReduceRotation((g * i) * (1 << (s * layersCollapse)), M / 4));
    }
  }

  if (flagRem) {
    int32_t s = levelBudget - flagRem;
    for (int32_t j = 0; j < gRem; j++) {
      indexList.emplace_back(ReduceRotation(
          (j - (numRotationsRem + 1) / 2 + 1) * (1 << (s * layersCollapse)),
          M / 4));
    }
    for (int32_t i = 0; i < bRem; i++) {
      indexList.emplace_back(
          ReduceRotation((gRem * i) * (1 << (s * layersCollapse)), M / 4));
    }
  }

  uint32_t m = slots * 4;
  if (m != M) {
    for (uint32_t j = 1; j < M / m; j <<= 1) {
      indexList.emplace_back(j * slots);
    }
  }

  return indexList;
}

std::vector<int32_t> FindLinearTransformRotationIndices(uint32_t slots,
                                                        uint32_t M) {
  std::vector<int32_t> indexList;

  for (int32_t i = 1; i < static_cast<int32_t>(slots); i++) {
    indexList.push_back(i);
  }

  uint32_t m = slots * 4;
  if (m != M) {
    for (uint32_t j = 1; j < M / m; j <<= 1) {
      indexList.emplace_back(j * slots);
    }
  }

  return indexList;
}

std::vector<int32_t> FindBootstrapRotationIndices(
    uint32_t slots, uint32_t M, const BootstrapParams& encParams,
    const BootstrapParams& decParams) {
  std::vector<int32_t> fullIndexList;

  bool isLTBootstrap =
      (encParams.levelBudget == 1) && (decParams.levelBudget == 1);

  if (isLTBootstrap) {
    fullIndexList = FindLinearTransformRotationIndices(slots, M);
  } else {
    fullIndexList = FindCoeffsToSlotsRotationIndices(slots, M, encParams);
    std::vector<int32_t> indexListStC =
        FindSlotsToCoeffsRotationIndices(slots, M, decParams);
    fullIndexList.insert(fullIndexList.end(), indexListStC.begin(),
                         indexListStC.end());
  }

  std::sort(fullIndexList.begin(), fullIndexList.end());
  fullIndexList.erase(std::unique(fullIndexList.begin(), fullIndexList.end()),
                      fullIndexList.end());

  fullIndexList.erase(
      std::remove(fullIndexList.begin(), fullIndexList.end(), 0),
      fullIndexList.end());
  fullIndexList.erase(std::remove(fullIndexList.begin(), fullIndexList.end(),
                                  static_cast<int32_t>(M / 4)),
                      fullIndexList.end());

  return fullIndexList;
}

std::vector<int32_t> CalculateBootstrapRotationIndices(
    uint32_t slots, uint32_t cyclotomicOrder,
    const std::vector<uint32_t>& levelBudget,
    const std::vector<uint32_t>& dim1) {
  if (levelBudget.size() != 2 || dim1.size() != 2) {
    return {};
  }

  if (slots == 0) {
    slots = cyclotomicOrder / 4;
  }

  uint32_t logSlots = static_cast<uint32_t>(std::log2(slots));
  if (logSlots == 0) {
    logSlots = 1;
  }

  std::vector<uint32_t> adjustedBudget = levelBudget;

  if (adjustedBudget[0] > logSlots) adjustedBudget[0] = logSlots;
  if (adjustedBudget[0] < 1) adjustedBudget[0] = 1;
  if (adjustedBudget[1] > logSlots) adjustedBudget[1] = logSlots;
  if (adjustedBudget[1] < 1) adjustedBudget[1] = 1;

  if (BootstrapKeyConfigRegistry::hasConfiguration(adjustedBudget[0],
                                                   adjustedBudget[1])) {
    std::string configStr = BootstrapKeyConfigRegistry::getConfigString(
        adjustedBudget[0], adjustedBudget[1]);
    llvm::dbgs() << "Using pre-analyzed bootstrap key levels for " << configStr
                 << " configuration\n";
    return BootstrapKeyConfigRegistry::getRotationIndices(adjustedBudget[0],
                                                          adjustedBudget[1]);
  }
  llvm::dbgs()
      << "No pre-analyzed configuration found, using algorithmic calculation\n";
  BootstrapParams encParams =
      GetCollapsedFFTParams(slots, adjustedBudget[0], dim1[0]);
  BootstrapParams decParams =
      GetCollapsedFFTParams(slots, adjustedBudget[1], dim1[1]);

  return FindBootstrapRotationIndices(slots, cyclotomicOrder, encParams,
                                      decParams);
}

#define GEN_PASS_DEF_BOOTSTRAPROTATIONANALYSIS
#include "lib/Transforms/BootstrapRotationAnalysis/BootstrapRotationAnalysis.h.inc"

struct BootstrapRotationAnalysis
    : impl::BootstrapRotationAnalysisBase<BootstrapRotationAnalysis> {
  void runOnOperation() override {
    Operation* op = getOperation();
    static bool initialized = false;
    if (!initialized) {
      BootstrapKeyConfigInitializer::initializeConfigurations();
      initialized = true;
    }

    // Find SetupBootstrapOp to extract parameters
    openfhe::SetupBootstrapOp setupOp = nullptr;
    op->walk([&](openfhe::SetupBootstrapOp foundOp) {
      setupOp = foundOp;
      return WalkResult::interrupt();
    });

    if (!setupOp) {
      // No bootstrap setup found, nothing to do
      return;
    }

    // Extract parameters from SetupBootstrapOp
    uint32_t slots = 0;
    uint32_t cyclotomicOrder = 0;
    std::vector<uint32_t> levelBudget = {3, 3};  // Default values
    std::vector<uint32_t> dim1 = {0, 0};         // Default values

    // Extract level budgets from SetupBootstrapOp attributes
    if (auto levelBudgetEncodeAttr =
            setupOp->getAttrOfType<IntegerAttr>("levelBudgetEncode")) {
      levelBudget[0] = static_cast<uint32_t>(levelBudgetEncodeAttr.getInt());
      llvm::dbgs() << "Found levelBudgetEncode: " << levelBudget[0] << "\n";
    }
    if (auto levelBudgetDecodeAttr =
            setupOp->getAttrOfType<IntegerAttr>("levelBudgetDecode")) {
      levelBudget[1] = static_cast<uint32_t>(levelBudgetDecodeAttr.getInt());
      llvm::dbgs() << "Found levelBudgetDecode: " << levelBudget[1] << "\n";
    }

    // Extract dim1 parameters if they exist (they might be optional)
    if (auto dim1EncodeAttr =
            setupOp->getAttrOfType<IntegerAttr>("dim1Encode")) {
      dim1[0] = static_cast<uint32_t>(dim1EncodeAttr.getInt());
      llvm::dbgs() << "Found dim1Encode: " << dim1[0] << "\n";
    }
    if (auto dim1DecodeAttr =
            setupOp->getAttrOfType<IntegerAttr>("dim1Decode")) {
      dim1[1] = static_cast<uint32_t>(dim1DecodeAttr.getInt());
      llvm::dbgs() << "Found dim1Decode: " << dim1[1] << "\n";
    }

    // Extract slots and cyclotomic order from context or use defaults
    // These might be stored elsewhere in the crypto context or as module
    // attributes
    if (auto slotsAttr = setupOp->getAttrOfType<IntegerAttr>("slots")) {
      slots = static_cast<uint32_t>(slotsAttr.getInt());
      llvm::dbgs() << "Found slots: " << slots << "\n";
    } else {
      slots = 1 << 15;  // Common default
      llvm::dbgs() << "Using default slots: " << slots << "\n";
    }

    if (auto cyclotomicAttr =
            setupOp->getAttrOfType<IntegerAttr>("cyclotomic_order")) {
      cyclotomicOrder = static_cast<uint32_t>(cyclotomicAttr.getInt());
      llvm::dbgs() << "Found cyclotomic_order: " << cyclotomicOrder << "\n";
    } else {
      cyclotomicOrder = 1 << 17;  // Common default for 4096 slots
      llvm::dbgs() << "Using default cyclotomic_order: " << cyclotomicOrder
                   << "\n";
    }

    // Calculate rotation indices
    std::vector<int32_t> rotationIndices = CalculateBootstrapRotationIndices(
        slots, cyclotomicOrder, levelBudget, dim1);

    // Convert to MLIR attributes
    OpBuilder builder(op->getContext());
    SmallVector<Attribute> indexAttrs;
    for (int32_t idx : rotationIndices) {
      indexAttrs.push_back(builder.getI32IntegerAttr(idx));
    }
    ArrayAttr rotationIndicesAttr = builder.getArrayAttr(indexAttrs);

    // Transform all bootstrap operations to include key management
    SmallVector<openfhe::BootstrapOp> bootstrapOps;
    op->walk([&](openfhe::BootstrapOp bootstrapOp) {
      bootstrapOps.push_back(bootstrapOp);
      return WalkResult::advance();
    });
    bool hasPreAnalyzedConfig = BootstrapKeyConfigRegistry::hasConfiguration(
        levelBudget[0], levelBudget[1]);
    std::string configStr = BootstrapKeyConfigRegistry::getConfigString(
        levelBudget[0], levelBudget[1]);

    // Process each bootstrap operation
    for (auto bootstrapOp : bootstrapOps) {
      transformBootstrapOp(bootstrapOp, rotationIndices, rotationIndicesAttr,
                           slots, cyclotomicOrder, builder, levelBudget[0],
                           levelBudget[1], hasPreAnalyzedConfig);
    }

    op->walk([&](openfhe::GenRotKeyOp genRotKeyOp) {
      // Mark that bootstrap is enabled for this rotation key generation
      genRotKeyOp->setAttr("bootstrap_enabled", builder.getBoolAttr(true));

      llvm::dbgs() << "Marked GenRotKeyOp with bootstrap_enabled = true\n";
      return WalkResult::advance();
    });
    // Add rotation indices to bootstrap key generation operations
    op->walk([&](openfhe::GenBootstrapKeyOp genBootstrapKeyOp) {
      genBootstrapKeyOp->setAttr("rotation_indices", rotationIndicesAttr);
      genBootstrapKeyOp->setAttr(
          "num_rotation_indices",
          builder.getI32IntegerAttr(rotationIndices.size()));
      genBootstrapKeyOp->setAttr("slots", builder.getI32IntegerAttr(slots));
      genBootstrapKeyOp->setAttr("cyclotomic_order",
                                 builder.getI32IntegerAttr(cyclotomicOrder));

      llvm::dbgs() << "Added rotation indices to GenBootstrapKeyOp:\n";
      llvm::dbgs() << "  " << rotationIndices.size()
                   << " rotation keys will be generated\n";
      llvm::dbgs() << "  Indices: [";
      for (size_t i = 0;
           i < std::min(static_cast<size_t>(10), rotationIndices.size()); ++i) {
        if (i > 0) llvm::dbgs() << ", ";
        llvm::dbgs() << rotationIndices[i];
      }
      if (rotationIndices.size() > 10) {
        llvm::dbgs() << ", ...";
      }
      llvm::dbgs() << "]\n";

      return WalkResult::advance();
    });

    // Print summary
    llvm::dbgs() << "Bootstrap Rotation Analysis:\n";
    llvm::dbgs() << "  Configuration: " << configStr << "\n";
    if (hasPreAnalyzedConfig) {
      llvm::dbgs() << "  Using pre-analyzed bootstrap key levels\n";
      auto distribution = BootstrapKeyConfigRegistry::getLevelDistribution(
          levelBudget[0], levelBudget[1]);
      for (const auto& entry : distribution) {
        llvm::dbgs() << "    Level " << entry.first << ": " << entry.second
                     << " keys\n";
      }
    } else {
      llvm::dbgs() << "  Using algorithmic calculation\n";
    }
    llvm::dbgs() << "  Total Rotation Indices: " << rotationIndices.size()
                 << "\n";
    llvm::dbgs() << "  Processed " << bootstrapOps.size()
                 << " bootstrap operations\n";
  }

 private:
  // Transform a bootstrap operation to include key loading and clearing
  void transformBootstrapOp(openfhe::BootstrapOp bootstrapOp,
                            const std::vector<int32_t>& rotationIndices,
                            ArrayAttr rotationIndicesAttr, uint32_t slots,
                            uint32_t cyclotomicOrder, OpBuilder& builder,
                            uint32_t encodeBudget, uint32_t decodeBudget,
                            bool hasPreAnalyzedConfig) {
    Location loc = bootstrapOp.getLoc();

    // Set insertion point before the bootstrap operation
    builder.setInsertionPoint(bootstrapOp);

    // Step 1: Load all rotation keys needed for bootstrap using KMRT operations
    SmallVector<Value> loadedKeys;
    SmallVector<kmrt::LoadKeyOp> loadOps;

    for (int32_t rotIndex : rotationIndices) {
      // Create constant for the rotation index
      auto indexConstant = builder.create<arith::ConstantOp>(
          loc, builder.getI64IntegerAttr(rotIndex));

      // Create load_key operation with static rotation index
      auto rotKeyType = kmrt::RotKeyType::get(builder.getContext(),
                                              std::optional<int64_t>(rotIndex));
      auto loadOp = builder.create<kmrt::LoadKeyOp>(loc, rotKeyType,
                                                    indexConstant.getResult());

      // For pre-analyzed configurations, set the correct level
      if (hasPreAnalyzedConfig && BootstrapKeyConfigRegistry::hasLevel(
                                      encodeBudget, decodeBudget, rotIndex)) {
        int level = BootstrapKeyConfigRegistry::getLevel(
            encodeBudget, decodeBudget, rotIndex);
        loadOp->setAttr("key_depth",
                        builder.getIntegerAttr(builder.getI64Type(), level));
        llvm::dbgs() << "Set bootstrap key level " << level
                     << " for rotation index " << rotIndex << "\n";
      }

      loadedKeys.push_back(loadOp.getResult());
      loadOps.push_back(loadOp);
    }

    // Step 2: Add attributes to the bootstrap operation
    bootstrapOp->setAttr("rotation_indices", rotationIndicesAttr);
    bootstrapOp->setAttr("num_rotation_indices",
                         builder.getI32IntegerAttr(rotationIndices.size()));
    bootstrapOp->setAttr("slots", builder.getI32IntegerAttr(slots));
    bootstrapOp->setAttr("cyclotomic_order",
                         builder.getI32IntegerAttr(cyclotomicOrder));

    // Step 3: Set insertion point after the bootstrap operation
    builder.setInsertionPointAfter(bootstrapOp);

    // Step 4: Clear all the rotation keys using KMRT clear_key
    for (Value key : loadedKeys) {
      builder.create<kmrt::ClearKeyOp>(loc, key);
    }

    // Print information about the transformation
    llvm::dbgs() << "Transformed bootstrap operation:\n";
    llvm::dbgs() << "  Added " << loadOps.size()
                 << " KMRT LoadKeyOp operations\n";
    llvm::dbgs() << "  Added " << loadedKeys.size()
                 << " KMRT ClearKeyOp operations\n";
    llvm::dbgs() << "  Rotation indices: [";
    for (size_t i = 0; i < rotationIndices.size(); ++i) {
      if (i > 0) llvm::dbgs() << ", ";
      llvm::dbgs() << rotationIndices[i];
      if (i >= 9) {  // Limit output for readability
        llvm::dbgs() << ", ...";
        break;
      }
    }
    llvm::dbgs() << "]\n";
  }
};

}  // namespace heir
}  // namespace mlir
