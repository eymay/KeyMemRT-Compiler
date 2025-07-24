#include "lib/Transforms/KeyCompression/KeyCompression.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <vector>

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Support/LLVM.h"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_KEYCOMPRESSION
#include "lib/Transforms/KeyCompression/KeyCompression.h.inc"

namespace {

// Profile data structure for rotation analysis
struct ProfileEntry {
  std::string inputMLIRName;
  unsigned level = 0;
  unsigned noise = 0;
  unsigned towers = 0;
  unsigned line = 0;
};

// Parser for rotation profile data
class ProfileParser {
 public:
  static std::vector<ProfileEntry> parseProfile(const std::string& filename) {
    std::vector<ProfileEntry> entries;
    std::ifstream file(filename);
    std::string line;

    if (!file.good()) {
      return entries;
    }

    while (std::getline(file, line)) {
      // Parse format:
      // "ROTATION_PROFILE::INPUT:%ct_915:LEVEL:1:NOISE:2:TOWERS:31:LINE:1077"
      if (line.find("ROTATION_PROFILE:") == 0) {
        auto entry = parseLine(line);
        if (!entry.inputMLIRName.empty()) {
          entries.push_back(entry);
        }
      }
    }

    return entries;
  }

 private:
  static ProfileEntry parseLine(const std::string& line) {
    ProfileEntry entry;

    // Split by colons
    std::vector<std::string> parts;
    std::stringstream ss(line);
    std::string part;

    while (std::getline(ss, part, ':')) {
      parts.push_back(part);
    }

    try {
      // Handle format:
      // ROTATION_PROFILE::INPUT:mlir_name:LEVEL:level:NOISE:noise:TOWERS:towers:LINE:line
      if (parts.size() >= 10 && parts[1].empty()) {
        entry.inputMLIRName = parts[3];
        entry.level = std::stoul(parts[5]);
        entry.noise = std::stoul(parts[7]);
        entry.towers = std::stoul(parts[9]);
        if (parts.size() >= 12) {
          entry.line = std::stoul(parts[11]);
        }
      }
    } catch (const std::exception&) {
      entry.inputMLIRName.clear();  // Mark as invalid
    }

    return entry;
  }
};

struct RotationUsage {
  openfhe::RotOp rotOp;
  unsigned level;
  std::string inputMLIRName;
};

struct BootstrapInfo {
  openfhe::BootstrapOp op;
  unsigned level;
  std::set<int32_t> rotationIndices;
  std::string inputMLIRName;
};

struct KeyCompression : impl::KeyCompressionBase<KeyCompression> {
  void runOnOperation() override {
    Operation* op = getOperation();
    MLIRContext* ctx = op->getContext();

    // Step 1: Load profile data
    auto profileEntries = ProfileParser::parseProfile("rotation_profile.txt");
    if (profileEntries.empty()) {
      llvm::errs() << "No profile data found. Please run the program first.\n";
      return;
    }

    llvm::errs() << "Loaded " << profileEntries.size() << " profile entries\n";

    // Step 2: Build mapping from MLIR input names to rotation operations
    std::map<std::string, int32_t> mlirInputToRotIndex;

    op->walk([&](openfhe::RotOp rotOp) {
      std::string inputMLIRName = getMLIRNameForValue(rotOp.getCiphertext());
      auto rotationIndex =
          rotOp.getEvalKey().getType().getRotationIndex().getInt();

      mlirInputToRotIndex[inputMLIRName] = rotationIndex;

      llvm::errs() << "Found rotation " << rotationIndex << " with input "
                   << inputMLIRName << "\n";
    });

    // Step 3: Analyze bootstrap operations and set levels for their keys
    analyzeBootstrapOperations(op, profileEntries, ctx);

    // Step 4: Handle regular rotation operations from profile data
    handleRegularRotations(op, profileEntries, mlirInputToRotIndex, ctx);

    // Step 5: Insert compress_key operations for keys used at multiple levels
    insertOptimalCompressionOperations(op, profileEntries, mlirInputToRotIndex,
                                       ctx);

    // Step 6: Generate GenRotKeyDepth operations based on deserialize levels
    generateGenRotKeyDepthOps(op, ctx);

    // Step 7: Enhanced report
    generateReport(op, profileEntries);
  }

 private:
  std::string getMLIRNameForValue(Value value) {
    std::string nameStr;
    llvm::raw_string_ostream nameStream(nameStr);
    if (auto parentOp = value.getParentRegion()->getParentOp()) {
      AsmState asmState(parentOp);
      value.printAsOperand(nameStream, asmState);
      nameStream.flush();
    }
    return nameStr;
  }

  void analyzeBootstrapOperations(
      Operation* op, const std::vector<ProfileEntry>& profileEntries,
      MLIRContext* ctx) {
    // Build a map of all deserialize operations by rotation index
    std::map<int32_t, std::vector<openfhe::DeserializeKeyOp>> allDeserializeOps;
    op->walk([&](openfhe::DeserializeKeyOp deserOp) {
      if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
        int32_t rotationIndex = indexAttr.getInt();
        allDeserializeOps[rotationIndex].push_back(deserOp);
      }
    });

    // Collect all bootstrap operations with their levels and rotation indices
    std::vector<BootstrapInfo> allBootstraps;

    op->walk([&](openfhe::BootstrapOp bootstrapOp) {
      std::string bootstrapInputMLIRName =
          getMLIRNameForValue(bootstrapOp.getCiphertext());

      // Find the level of the bootstrap input from profile data
      unsigned bootstrapInputLevel = 0;
      bool foundBootstrapLevel = false;

      for (const auto& entry : profileEntries) {
        if (entry.inputMLIRName == bootstrapInputMLIRName) {
          bootstrapInputLevel = entry.level;
          foundBootstrapLevel = true;
          break;
        }
      }

      if (!foundBootstrapLevel) {
        llvm::errs() << "Warning: No profile data found for bootstrap input "
                     << bootstrapInputMLIRName << "\n";
        return;
      }

      // Get rotation indices used by this bootstrap
      std::set<int32_t> bootstrapRotationIndices;
      if (auto rotationIndicesAttr =
              bootstrapOp->getAttrOfType<ArrayAttr>("rotation_indices")) {
        for (auto indexAttr : rotationIndicesAttr.getAsRange<IntegerAttr>()) {
          int32_t rotationIndex = indexAttr.getInt();
          bootstrapRotationIndices.insert(rotationIndex);
        }
      }

      BootstrapInfo info;
      info.op = bootstrapOp;
      info.level = bootstrapInputLevel;
      info.rotationIndices = bootstrapRotationIndices;
      info.inputMLIRName = bootstrapInputMLIRName;

      allBootstraps.push_back(info);

      llvm::errs() << "Found bootstrap at level " << bootstrapInputLevel
                   << " with " << bootstrapRotationIndices.size()
                   << " rotation indices\n";
    });

    // For each set of keys, find which bootstraps use them and handle
    // multi-level cases
    for (const auto& [rotationIndex, deserializeOps] : allDeserializeOps) {
      // Find all bootstraps that use this rotation index
      std::vector<BootstrapInfo> bootstrapsUsingThisKey;

      for (const auto& bootstrap : allBootstraps) {
        if (bootstrap.rotationIndices.count(rotationIndex)) {
          bootstrapsUsingThisKey.push_back(bootstrap);
        }
      }

      if (bootstrapsUsingThisKey.empty()) continue;

      // Sort bootstraps by program order
      std::sort(bootstrapsUsingThisKey.begin(), bootstrapsUsingThisKey.end(),
                [this](const BootstrapInfo& a, const BootstrapInfo& b) {
                  return isOperationBefore(
                      const_cast<openfhe::BootstrapOp&>(a.op).getOperation(),
                      const_cast<openfhe::BootstrapOp&>(b.op).getOperation());
                });

      // Handle the case where multiple bootstraps use the same keys
      handleMultiBootstrapKeyUsage(rotationIndex, bootstrapsUsingThisKey,
                                   deserializeOps, ctx);
    }
  }

  void handleMultiBootstrapKeyUsage(
      int32_t rotationIndex, const std::vector<BootstrapInfo>& bootstraps,
      const std::vector<openfhe::DeserializeKeyOp>& deserializeOps,
      MLIRContext* ctx) {
    if (bootstraps.size() == 1) {
      // Single bootstrap case - handle normally
      const auto& bootstrap = bootstraps[0];

      // Find the deserialize operation that is live at this bootstrap
      openfhe::DeserializeKeyOp liveDeserOp = nullptr;
      for (auto deserOp : deserializeOps) {
        if (isKeyLiveAtBootstrap(deserOp, bootstrap.op)) {
          liveDeserOp = deserOp;
          break;
        }
      }

      if (liveDeserOp) {
        setLevelForSpecificDeserializeOp(liveDeserOp, bootstrap.level,
                                         rotationIndex, ctx);
        llvm::errs() << "Single bootstrap: Set level " << bootstrap.level
                     << " for rotation " << rotationIndex << "\n";
      }
    } else {
      // Multiple bootstrap case - need compression between them
      llvm::errs() << "Multiple bootstraps (" << bootstraps.size()
                   << ") use rotation " << rotationIndex << " at levels: ";
      for (const auto& bootstrap : bootstraps) {
        llvm::errs() << bootstrap.level << " ";
      }
      llvm::errs() << "\n";

      // Find the deserialize operation(s) that cover all these bootstraps
      std::vector<openfhe::DeserializeKeyOp> liveDeserOps;
      for (auto deserOp : deserializeOps) {
        bool coversAllBootstraps = true;
        for (const auto& bootstrap : bootstraps) {
          if (!isKeyLiveAtBootstrap(deserOp, bootstrap.op)) {
            coversAllBootstraps = false;
            break;
          }
        }
        if (coversAllBootstraps) {
          liveDeserOps.push_back(deserOp);
        }
      }

      if (liveDeserOps.empty()) {
        llvm::errs()
            << "Warning: No deserialize op covers all bootstraps for rotation "
            << rotationIndex << "\n";
        return;
      }

      // Use the first deserialize that covers all bootstraps
      auto primaryDeserOp = liveDeserOps[0];

      // Set deserialize level to the minimum level needed
      unsigned minLevel = bootstraps[0].level;
      for (const auto& bootstrap : bootstraps) {
        minLevel = std::min(minLevel, bootstrap.level);
      }

      setLevelForSpecificDeserializeOp(primaryDeserOp, minLevel, rotationIndex,
                                       ctx);
      llvm::errs() << "Multi-bootstrap: Set deserialize level " << minLevel
                   << " for rotation " << rotationIndex << "\n";

      // Insert compression operations between bootstraps
      insertCompressionBetweenBootstraps(rotationIndex, bootstraps, ctx);
    }
  }

  void insertCompressionBetweenBootstraps(
      int32_t rotationIndex, const std::vector<BootstrapInfo>& bootstraps,
      MLIRContext* ctx) {
    // Group bootstraps by level
    std::map<unsigned, std::vector<BootstrapInfo>> bootstrapsByLevel;
    for (const auto& bootstrap : bootstraps) {
      bootstrapsByLevel[bootstrap.level].push_back(bootstrap);
    }

    // Sort levels
    std::vector<unsigned> sortedLevels;
    for (const auto& [level, _] : bootstrapsByLevel) {
      sortedLevels.push_back(level);
    }
    std::sort(sortedLevels.begin(), sortedLevels.end());

    // Insert compression after each level (except the last)
    for (size_t i = 0; i < sortedLevels.size() - 1; i++) {
      unsigned currentLevel = sortedLevels[i];
      unsigned nextLevel = sortedLevels[i + 1];

      // Find the last bootstrap at the current level
      auto& bootstrapsAtLevel = bootstrapsByLevel[currentLevel];
      auto lastBootstrapAtLevel = bootstrapsAtLevel.back();

      // Insert compression after this bootstrap
      insertCompressionAfterBootstrap(lastBootstrapAtLevel.op, rotationIndex,
                                      nextLevel, ctx);

      llvm::errs() << "Inserted compression between bootstraps for rotation "
                   << rotationIndex << " from level " << currentLevel << " to "
                   << nextLevel << "\n";
    }
  }

  void insertCompressionAfterBootstrap(openfhe::BootstrapOp bootstrapOp,
                                       int32_t rotationIndex,
                                       unsigned targetLevel, MLIRContext* ctx) {
    OpBuilder builder(bootstrapOp);
    builder.setInsertionPointAfter(bootstrapOp);

    // We need to find the evaluation key for this rotation index
    // Look for deserialize operations before this bootstrap
    Value evalKeyToCompress = nullptr;

    Operation* current = bootstrapOp.getOperation()->getPrevNode();
    while (current != nullptr) {
      if (auto deserOp = dyn_cast<openfhe::DeserializeKeyOp>(current)) {
        if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
          if (indexAttr.getInt() == rotationIndex) {
            evalKeyToCompress = deserOp.getResult();
            break;
          }
        }
      }
      current = current->getPrevNode();
    }

    if (!evalKeyToCompress) {
      llvm::errs() << "Warning: Could not find eval key for rotation "
                   << rotationIndex << " to compress after bootstrap\n";
      return;
    }

    // Create compress_key operation
    auto compressOp = builder.create<openfhe::CompressKeyOp>(
        bootstrapOp.getLoc(), evalKeyToCompress.getType(),
        bootstrapOp.getCryptoContext(), evalKeyToCompress,
        builder.getI32IntegerAttr(targetLevel));

    // Add metadata
    compressOp->setAttr("bootstrap_compression", BoolAttr::get(ctx, true));
    compressOp->setAttr(
        "target_level",
        IntegerAttr::get(IntegerType::get(ctx, 32), targetLevel));
    compressOp->setAttr(
        "rotation_index",
        IntegerAttr::get(IntegerType::get(ctx, 32), rotationIndex));

    llvm::errs() << "Inserted compression after bootstrap for rotation "
                 << rotationIndex << " to level " << targetLevel << "\n";
  }

  void handleRegularRotations(
      Operation* op, const std::vector<ProfileEntry>& profileEntries,
      const std::map<std::string, int32_t>& mlirInputToRotIndex,
      MLIRContext* ctx) {
    op->walk([&](openfhe::DeserializeKeyOp deserOp) {
      // Check if this specific deserialize operation has been assigned a level
      // by bootstrap analysis
      if (deserOp->hasAttr("assigned_key_depth")) {
        return;  // Already processed
      }

      // Handle regular rotations (non-bootstrap)
      int32_t rotationIndex = -1;
      if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
        rotationIndex = indexAttr.getInt();
      }

      if (rotationIndex == -1) {
        return;
      }

      // Check if we have profile data for regular rotations using this key
      unsigned minLevel = UINT_MAX;
      bool foundRegularUsage = false;

      for (const auto& entry : profileEntries) {
        auto it = mlirInputToRotIndex.find(entry.inputMLIRName);
        if (it != mlirInputToRotIndex.end() && it->second == rotationIndex) {
          minLevel = std::min(minLevel, entry.level);
          foundRegularUsage = true;
        }
      }

      if (foundRegularUsage) {
        deserOp->setAttr("key_depth",
                         IntegerAttr::get(IntegerType::get(ctx, 64), minLevel));
        llvm::errs() << "Set key_depth=" << minLevel << " for regular rotation "
                     << rotationIndex << "\n";
      }
    });
  }

  void insertOptimalCompressionOperations(
      Operation* op, const std::vector<ProfileEntry>& profileEntries,
      const std::map<std::string, int32_t>& mlirInputToRotIndex,
      MLIRContext* ctx) {
    // Analyze which rotation indices are used at multiple levels
    std::map<int32_t, std::set<unsigned>> rotationToLevelsUsed;
    std::map<int32_t, std::vector<RotationUsage>> rotationToUsages;

    // Collect all rotation usages with their operations
    op->walk([&](openfhe::RotOp rotOp) {
      auto rotationIndex =
          rotOp.getEvalKey().getType().getRotationIndex().getInt();
      std::string inputMLIRName = getMLIRNameForValue(rotOp.getCiphertext());

      // Find the level for this usage from profile data
      for (const auto& entry : profileEntries) {
        if (entry.inputMLIRName == inputMLIRName) {
          RotationUsage usage;
          usage.rotOp = rotOp;
          usage.level = entry.level;
          usage.inputMLIRName = inputMLIRName;

          rotationToLevelsUsed[rotationIndex].insert(entry.level);
          rotationToUsages[rotationIndex].push_back(usage);
          break;
        }
      }
    });

    // Find rotations that need compression (used at multiple levels)
    for (const auto& [rotationIndex, levelsUsed] : rotationToLevelsUsed) {
      if (levelsUsed.size() <= 1) {
        continue;  // Single level usage - no compression needed
      }

      unsigned deserializeLevel =
          *std::min_element(levelsUsed.begin(), levelsUsed.end());
      auto& usages = rotationToUsages[rotationIndex];

      // Sort usages by level to find optimal compression points
      std::sort(usages.begin(), usages.end(),
                [](const RotationUsage& a, const RotationUsage& b) {
                  return a.level < b.level;
                });

      llvm::errs() << "Rotation " << rotationIndex << " needs compression: "
                   << "deserialize at " << deserializeLevel
                   << ", used at levels: ";
      for (auto level : levelsUsed) {
        llvm::errs() << level << " ";
      }
      llvm::errs() << "\n";

      // Insert compression operations at optimal points
      insertOptimalCompressionForRotation(rotationIndex, usages,
                                          deserializeLevel, ctx);
    }
  }

  void insertOptimalCompressionForRotation(
      int32_t rotationIndex, const std::vector<RotationUsage>& usages,
      unsigned deserializeLevel, MLIRContext* ctx) {
    // Group usages by level
    std::map<unsigned, std::vector<RotationUsage>> usagesByLevel;
    for (const auto& usage : usages) {
      usagesByLevel[usage.level].push_back(usage);
    }

    // Process levels in ascending order
    std::vector<unsigned> sortedLevels;
    for (const auto& [level, _] : usagesByLevel) {
      sortedLevels.push_back(level);
    }
    std::sort(sortedLevels.begin(), sortedLevels.end());

    Value currentKey = nullptr;  // Track the current key version

    for (size_t i = 0; i < sortedLevels.size(); i++) {
      unsigned currentLevel = sortedLevels[i];
      auto& levelUsages = usagesByLevel[currentLevel];

      // Update all rotations at this level to use the current key
      for (auto& usage : levelUsages) {
        if (currentKey && currentLevel > deserializeLevel) {
          // Use compressed key for higher levels
          usage.rotOp.getEvalKeyMutable().assign(currentKey);
        }
        // For the first level (deserialize level), operations use the original
        // key
      }

      // After processing all operations at this level, compress for the next
      // level if needed
      if (i + 1 < sortedLevels.size()) {
        unsigned nextLevel = sortedLevels[i + 1];

        // Find the last operation at the current level to insert compression
        // after it
        openfhe::RotOp lastOpAtLevel = levelUsages.back().rotOp;

        // Insert compression operation after the last usage at current level
        currentKey = insertCompressionAfterOperation(lastOpAtLevel, nextLevel,
                                                     rotationIndex, ctx);

        llvm::errs() << "Inserted compression for rotation " << rotationIndex
                     << " from level " << currentLevel << " to " << nextLevel
                     << " after last usage at level " << currentLevel << "\n";
      }
    }
  }

  Value insertCompressionAfterOperation(openfhe::RotOp rotOp,
                                        unsigned targetLevel,
                                        int32_t rotationIndex,
                                        MLIRContext* ctx) {
    OpBuilder builder(rotOp);
    builder.setInsertionPointAfter(rotOp);

    // Create compress_key operation right after the rotation
    auto compressOp = builder.create<openfhe::CompressKeyOp>(
        rotOp.getLoc(), rotOp.getEvalKey().getType(), rotOp.getCryptoContext(),
        rotOp.getEvalKey(),  // Compress the original key
        builder.getI32IntegerAttr(targetLevel));

    // Add metadata for tracking
    compressOp->setAttr("profile_guided", BoolAttr::get(ctx, true));
    compressOp->setAttr(
        "target_level",
        IntegerAttr::get(IntegerType::get(ctx, 32), targetLevel));
    compressOp->setAttr(
        "rotation_index",
        IntegerAttr::get(IntegerType::get(ctx, 32), rotationIndex));
    compressOp->setAttr("optimal_placement", BoolAttr::get(ctx, true));

    return compressOp.getResult();
  }

  void generateGenRotKeyDepthOps(Operation* op, MLIRContext* ctx) {
    // Collect rotation indices grouped by level from deserialize operations
    std::map<unsigned, std::vector<int32_t>> levelToRotationIndices;

    op->walk([&](openfhe::DeserializeKeyOp deserOp) {
      if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
        int32_t rotationIndex = indexAttr.getInt();

        unsigned level = 0;  // Default level
        if (auto keyDepthAttr =
                deserOp->getAttrOfType<IntegerAttr>("key_depth")) {
          level = keyDepthAttr.getInt();
        }

        levelToRotationIndices[level].push_back(rotationIndex);
        llvm::errs() << "Collected rotation " << rotationIndex << " at level "
                     << level << "\n";
      }
    });

    // Find existing GenRotKeyOp to replace
    openfhe::GenRotKeyOp existingGenRotKeyOp = nullptr;
    op->walk([&](openfhe::GenRotKeyOp genRotKeyOp) {
      if (!existingGenRotKeyOp) {
        existingGenRotKeyOp = genRotKeyOp;
      }
    });

    if (!existingGenRotKeyOp) {
      llvm::errs() << "Warning: No existing GenRotKeyOp found to replace\n";
      return;
    }

    // Create GenRotKeyDepthOp for each level
    OpBuilder builder(existingGenRotKeyOp);

    for (const auto& [level, rotationIndices] : levelToRotationIndices) {
      if (rotationIndices.empty()) continue;

      // Remove duplicates and sort
      std::set<int32_t> uniqueIndices(rotationIndices.begin(),
                                      rotationIndices.end());
      std::vector<int32_t> sortedIndices(uniqueIndices.begin(),
                                         uniqueIndices.end());

      // Create array attribute for rotation indices
      std::vector<Attribute> indexAttrs;
      for (int32_t index : sortedIndices) {
        indexAttrs.push_back(
            IntegerAttr::get(IntegerType::get(ctx, 64), index));
      }
      ArrayAttr rotationIndicesAttr = ArrayAttr::get(ctx, indexAttrs);

      // Create GenRotKeyDepthOp
      builder.create<openfhe::GenRotKeyDepthOp>(
          existingGenRotKeyOp.getLoc(), existingGenRotKeyOp.getCryptoContext(),
          existingGenRotKeyOp.getPrivateKey(), rotationIndicesAttr,
          IntegerAttr::get(IntegerType::get(ctx, 64), level));

      llvm::errs() << "Created GenRotKeyDepthOp for level " << level << " with "
                   << sortedIndices.size() << " rotation indices\n";
    }

    // Remove the original GenRotKeyOp
    existingGenRotKeyOp.erase();
    llvm::errs() << "Removed original GenRotKeyOp\n";
  }

  void generateReport(Operation* op,
                      const std::vector<ProfileEntry>& profileEntries) {
    llvm::errs() << "\n=== Key Compression Report ===\n";
    llvm::errs() << "Profile entries processed: " << profileEntries.size()
                 << "\n";

    // Count deserialize operations by level
    std::map<unsigned, int> levelCounts;
    op->walk([&](openfhe::DeserializeKeyOp deserOp) {
      if (auto keyDepthAttr =
              deserOp->getAttrOfType<IntegerAttr>("key_depth")) {
        unsigned level = keyDepthAttr.getInt();
        levelCounts[level]++;
      }
    });

    // Count compression operations
    int compressionCount = 0;
    op->walk([&](openfhe::CompressKeyOp compressOp) {
      if (compressOp->hasAttr("profile_guided")) {
        compressionCount++;
      }
    });

    llvm::errs() << "\nDeserialize operations by level:\n";
    for (const auto& [level, count] : levelCounts) {
      llvm::errs() << "  Level " << level << ": " << count << " keys\n";
    }

    llvm::errs() << "\nProfile-guided compression operations: "
                 << compressionCount << "\n";
    llvm::errs() << "=== End Report ===\n";
  }

  // Helper functions for liveness analysis
  bool isKeyLiveAtBootstrap(openfhe::DeserializeKeyOp deserOp,
                            openfhe::BootstrapOp bootstrapOp) {
    Value evalKey = deserOp.getResult();
    Operation* deserOperation = deserOp.getOperation();
    Operation* bootstrapOperation = bootstrapOp.getOperation();

    // Check if deserialize comes before bootstrap
    if (!isOperationBefore(deserOperation, bootstrapOperation)) {
      return false;  // Deserialize happens after bootstrap
    }

    // Check if the key is cleared before the bootstrap
    return !isKeyClearedBetween(evalKey, deserOperation, bootstrapOperation);
  }

  void setLevelForSpecificDeserializeOp(openfhe::DeserializeKeyOp deserOp,
                                        unsigned level, int32_t rotationIndex,
                                        MLIRContext* ctx) {
    deserOp->setAttr("key_depth",
                     IntegerAttr::get(IntegerType::get(ctx, 64), level));
    deserOp->setAttr("assigned_key_depth",
                     BoolAttr::get(ctx, true));  // Mark as processed
  }

  bool isOperationBefore(Operation* op1, Operation* op2) {
    if (!op1 || !op2) return false;

    // If they're in the same block, compare positions
    if (op1->getBlock() == op2->getBlock()) {
      Operation* current = op1;
      while (current != nullptr) {
        if (current == op2) {
          return true;  // op2 comes after op1
        }
        current = current->getNextNode();
      }
      return false;  // op2 comes before op1 or not found
    }

    return false;  // Different blocks - assume not ordered for now
  }

  bool isKeyClearedBetween(Value evalKey, Operation* startOp,
                           Operation* endOp) {
    Operation* current = startOp->getNextNode();

    while (current != nullptr && current != endOp) {
      if (auto clearOp = dyn_cast<openfhe::ClearKeyOp>(current)) {
        if (clearOp.getEvalKey() == evalKey) {
          return true;  // Key is cleared
        }
      }
      current = current->getNextNode();
    }

    return false;  // Key is not cleared
  }
};

}  // namespace

}  // namespace heir
}  // namespace mlir
