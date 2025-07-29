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
  using impl::KeyCompressionBase<KeyCompression>::KeyCompressionBase;
  void runOnOperation() override {
    Operation* op = getOperation();
    MLIRContext* ctx = op->getContext();

    // Step 1: Load profile data
    std::string profileFile = profileFilePath.getValue();
    auto profileEntries = ProfileParser::parseProfile(profileFile);
    if (profileEntries.empty()) {
      llvm::errs() << "No profile data found. "
                      "Please run the program first.\n";
      return;
    }

    llvm::errs() << "Loaded " << profileEntries.size() << " profile entries\n";

    // Step 2: Build mapping from ciphertext MLIR names to their levels
    std::map<std::string, unsigned> ciphertextToLevel;

    for (const auto& entry : profileEntries) {
      ciphertextToLevel[entry.inputMLIRName] = entry.level;
      llvm::errs() << "Ciphertext " << entry.inputMLIRName << " is at level "
                   << entry.level << "\n";
    }

    // Step 3: For each rotation operation, find the level of its input
    // ciphertext and set the corresponding deserialize key to match that level
    op->walk([&](openfhe::RotOp rotOp) {
      std::string inputMLIRName = getMLIRNameForValue(rotOp.getCiphertext());
      auto rotationIndex =
          rotOp.getEvalKey().getType().getRotationIndex().getInt();

      auto it = ciphertextToLevel.find(inputMLIRName);
      if (it != ciphertextToLevel.end()) {
        unsigned ciphertextLevel = it->second;

        // Find the deserialize operation that provides the key for this
        // rotation
        Value evalKey = rotOp.getEvalKey();
        Operation* deserOp = evalKey.getDefiningOp();

        if (auto deserializeOp = dyn_cast<openfhe::DeserializeKeyOp>(deserOp)) {
          // Set the key depth to match the ciphertext level
          deserializeOp->setAttr(
              "key_depth",
              IntegerAttr::get(IntegerType::get(ctx, 64), ciphertextLevel));

          llvm::errs() << "Set key_depth=" << ciphertextLevel
                       << " for rotation " << rotationIndex
                       << " (matches ciphertext " << inputMLIRName
                       << " level)\n";
        } else {
          llvm::errs() << "Warning: Could not find deserialize op for rotation "
                       << rotationIndex << "\n";
        }
      } else {
        llvm::errs() << "Warning: No level found for ciphertext "
                     << inputMLIRName << " used by rotation " << rotationIndex
                     << "\n";
      }
    });

    // Step 4: DISABLED - Skip compression operations
    // insertOptimalCompressionOperations(op, profileEntries,
    // mlirInputToRotIndex, ctx);

    // Step 5: Generate GenRotKeyDepth operations based on deserialize levels
    generateGenRotKeyDepthOps(op, ctx);

    // Step 6: Enhanced report
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
      // Find all bootstraps that use this rotation index AND share the same key
      // lifetime
      std::vector<BootstrapInfo> bootstrapsUsingThisKey;

      // Get the key lifetime for this rotation index
      for (auto deserOp : deserializeOps) {
        auto keyLifetimeBootstraps =
            getBootstrapsInKeyLifetime(deserOp, allBootstraps, rotationIndex);

        // Add these bootstraps to our list if they're not already there
        for (const auto& bootstrap : keyLifetimeBootstraps) {
          // Check if we already have this bootstrap
          bool alreadyAdded = false;
          for (const auto& existing : bootstrapsUsingThisKey) {
            if (existing.op == bootstrap.op) {
              alreadyAdded = true;
              break;
            }
          }
          if (!alreadyAdded) {
            bootstrapsUsingThisKey.push_back(bootstrap);
          }
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

  // NEW: Get bootstraps that occur within the lifetime of a specific
  // deserialize key
  std::vector<BootstrapInfo> getBootstrapsInKeyLifetime(
      openfhe::DeserializeKeyOp deserOp,
      const std::vector<BootstrapInfo>& allBootstraps, int32_t rotationIndex) {
    std::vector<BootstrapInfo> bootstrapsInLifetime;

    // Find the clear operation for this deserialize
    Value evalKey = deserOp.getResult();
    Operation* deserOperation = deserOp.getOperation();
    Operation* clearOperation = nullptr;

    // Look for the clear operation for this key
    Operation* current = deserOperation->getNextNode();
    while (current != nullptr) {
      if (auto clearOp = dyn_cast<openfhe::ClearKeyOp>(current)) {
        if (clearOp.getEvalKey() == evalKey) {
          clearOperation = current;
          break;
        }
      }
      current = current->getNextNode();
    }

    // Find all bootstraps that use this rotation index and occur between deser
    // and clear
    for (const auto& bootstrap : allBootstraps) {
      if (bootstrap.rotationIndices.count(rotationIndex)) {
        Operation* bootstrapOp =
            const_cast<openfhe::BootstrapOp&>(bootstrap.op).getOperation();

        // Check if bootstrap is between deserialize and clear (or after
        // deserialize if no clear)
        bool isAfterDeser = isOperationBefore(deserOperation, bootstrapOp);
        bool isBeforeClear = (clearOperation == nullptr) ||
                             isOperationBefore(bootstrapOp, clearOperation);

        if (isAfterDeser && isBeforeClear) {
          bootstrapsInLifetime.push_back(bootstrap);
        }
      }
    }

    return bootstrapsInLifetime;
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

      // Sort usages by program order (not by level) to maintain dominance
      std::sort(usages.begin(), usages.end(),
                [this](const RotationUsage& a, const RotationUsage& b) {
                  return isOperationBefore(
                      const_cast<openfhe::RotOp&>(a.rotOp).getOperation(),
                      const_cast<openfhe::RotOp&>(b.rotOp).getOperation());
                });

      llvm::errs() << "Rotation " << rotationIndex << " needs compression: "
                   << "deserialize at " << deserializeLevel
                   << ", used at levels: ";
      for (auto level : levelsUsed) {
        llvm::errs() << level << " ";
      }
      llvm::errs() << "\n";

      // Insert compression operations preserving dominance
      insertCompressionWithDominancePreservation(rotationIndex, usages, ctx);
    }
  }

  void insertCompressionWithDominancePreservation(
      int32_t rotationIndex, const std::vector<RotationUsage>& usages,
      MLIRContext* ctx) {
    // Build a map of deserialize→clear ranges for this rotation index
    std::vector<KeyLifetimeRange> keyRanges =
        findKeyLifetimeRanges(rotationIndex);

    if (keyRanges.empty()) {
      llvm::errs() << "Warning: No key lifetime ranges found for rotation "
                   << rotationIndex << "\n";
      return;
    }

    // Group usages by level, maintaining program order within each level
    std::map<unsigned, std::vector<RotationUsage>> usagesByLevel;
    for (const auto& usage : usages) {
      usagesByLevel[usage.level].push_back(usage);
    }

    // Sort levels in ascending order
    std::vector<unsigned> sortedLevels;
    for (const auto& [level, _] : usagesByLevel) {
      sortedLevels.push_back(level);
    }
    std::sort(sortedLevels.begin(), sortedLevels.end());

    // Process each key lifetime range
    for (auto& keyRange : keyRanges) {
      processKeyLifetimeRange(keyRange, usagesByLevel, sortedLevels,
                              rotationIndex, ctx);
    }
  }

  struct KeyLifetimeRange {
    openfhe::DeserializeKeyOp deserializeOp;
    openfhe::ClearKeyOp clearOp;  // can be null if no clear found
    Value keyValue;
    std::vector<openfhe::RotOp> rotationsInRange;
  };

  std::vector<KeyLifetimeRange> findKeyLifetimeRanges(int32_t rotationIndex) {
    std::vector<KeyLifetimeRange> ranges;

    // Find all deserialize operations for this rotation index
    getOperation()->walk([&](openfhe::DeserializeKeyOp deserOp) {
      if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
        if (indexAttr.getInt() == rotationIndex) {
          KeyLifetimeRange range;
          range.deserializeOp = deserOp;
          range.keyValue = deserOp.getResult();
          range.clearOp = nullptr;  // Will be found below

          // Find the corresponding clear operation
          Operation* current = deserOp.getOperation()->getNextNode();
          while (current != nullptr) {
            if (auto clearOp = dyn_cast<openfhe::ClearKeyOp>(current)) {
              if (clearOp.getEvalKey() == range.keyValue) {
                range.clearOp = clearOp;
                break;
              }
            }
            current = current->getNextNode();
          }

          // Find all rotation operations that use this key in its lifetime
          current = deserOp.getOperation()->getNextNode();
          Operation* endOp =
              range.clearOp ? range.clearOp.getOperation() : nullptr;

          while (current != nullptr && current != endOp) {
            if (auto rotOp = dyn_cast<openfhe::RotOp>(current)) {
              if (rotOp.getEvalKey() == range.keyValue) {
                range.rotationsInRange.push_back(rotOp);
              }
            }
            current = current->getNextNode();
          }

          ranges.push_back(range);
        }
      }
    });

    return ranges;
  }

  void processKeyLifetimeRange(
      KeyLifetimeRange& keyRange,
      const std::map<unsigned, std::vector<RotationUsage>>& usagesByLevel,
      const std::vector<unsigned>& sortedLevels, int32_t rotationIndex,
      MLIRContext* ctx) {
    // Find which rotations in this range need compression
    std::map<unsigned, std::vector<openfhe::RotOp>> rotationsByLevel;

    for (auto rotOp : keyRange.rotationsInRange) {
      // Find the level for this rotation from the usages
      for (const auto& [level, usages] : usagesByLevel) {
        for (const auto& usage : usages) {
          if (const_cast<openfhe::RotOp&>(usage.rotOp).getOperation() ==
              rotOp.getOperation()) {
            rotationsByLevel[level].push_back(rotOp);
            break;
          }
        }
      }
    }

    if (rotationsByLevel.empty()) {
      return;  // No rotations in this range need compression
    }

    // Sort the levels that are actually used in this range
    std::vector<unsigned> levelsInRange;
    for (const auto& [level, _] : rotationsByLevel) {
      levelsInRange.push_back(level);
    }
    std::sort(levelsInRange.begin(), levelsInRange.end());

    if (levelsInRange.size() <= 1) {
      return;  // Single level, no compression needed
    }

    // Process levels sequentially, creating compressions within this key's
    // lifetime
    Value currentKey = keyRange.keyValue;  // Start with the deserialize key

    for (size_t i = 0; i < levelsInRange.size(); i++) {
      unsigned currentLevel = levelsInRange[i];
      auto& rotationsAtLevel = rotationsByLevel[currentLevel];

      // Update rotations at this level to use the current key (if not the
      // original)
      if (i > 0) {
        for (auto rotOp : rotationsAtLevel) {
          replaceRotationWithCompressedKey(rotOp, currentKey, ctx);
        }
      }

      // After the last rotation at this level, create compressed key for next
      // level
      if (i + 1 < levelsInRange.size()) {
        unsigned nextLevel = levelsInRange[i + 1];
        openfhe::RotOp lastRotAtLevel = rotationsAtLevel.back();

        // Create compressed key right after the last rotation at this level
        Value compressedKey = createCompressedKeyAfterRotation(
            lastRotAtLevel, currentKey, nextLevel, rotationIndex, ctx);

        if (compressedKey) {
          currentKey = compressedKey;

          llvm::errs() << "Created compressed key for rotation "
                       << rotationIndex << " from level " << currentLevel
                       << " to " << nextLevel << " within key lifetime range\n";
        }
      }
    }

    // Update the clear operation to clear the final compressed key instead of
    // original
    if (keyRange.clearOp && currentKey != keyRange.keyValue) {
      OpBuilder builder(keyRange.clearOp);

      // Create new clear operation for the compressed key
      builder.create<openfhe::ClearKeyOp>(
          keyRange.clearOp.getLoc(), keyRange.clearOp.getCryptoContext(),
          currentKey);  // Clear the compressed key instead

      // Remove the original clear operation
      keyRange.clearOp.erase();

      llvm::errs()
          << "Updated clear operation to clear compressed key for rotation "
          << rotationIndex << "\n";
    }
  }

  Value createCompressedKeyAfterRotation(openfhe::RotOp rotOp,
                                         Value keyToCompress,
                                         unsigned targetLevel,
                                         int32_t rotationIndex,
                                         MLIRContext* ctx) {
    // Verify inputs
    if (!keyToCompress) {
      llvm::errs() << "Warning: Cannot create compressed key from null key\n";
      return nullptr;
    }

    OpBuilder builder(rotOp);
    builder.setInsertionPointAfter(rotOp);

    // Create compress_key operation right after the rotation that used the
    // uncompressed key
    auto compressOp = builder.create<openfhe::CompressKeyOp>(
        rotOp.getLoc(), keyToCompress.getType(), rotOp.getCryptoContext(),
        keyToCompress,  // Compress the key that was just used
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

    llvm::errs() << "Created compress_key operation for rotation "
                 << rotationIndex << " to level " << targetLevel
                 << " after rotation usage\n";

    return compressOp.getResult();
  }

  Value findOriginalDeserializeKey(int32_t rotationIndex) {
    Value result = nullptr;

    getOperation()->walk([&](openfhe::DeserializeKeyOp deserOp) {
      if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
        if (indexAttr.getInt() == rotationIndex) {
          result = deserOp.getResult();
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    return result;
  }

  void replaceRotationWithCompressedKey(openfhe::RotOp originalRotOp,
                                        Value compressedKey, MLIRContext* ctx) {
    // Verify the compressed key is valid
    if (!compressedKey) {
      llvm::errs()
          << "Warning: Cannot replace rotation with null compressed key\n";
      return;
    }

    OpBuilder builder(originalRotOp);

    // Create a new rotation operation with the compressed key
    auto newRotOp = builder.create<openfhe::RotOp>(
        originalRotOp.getLoc(), originalRotOp.getResult().getType(),
        originalRotOp.getCryptoContext(), originalRotOp.getCiphertext(),
        compressedKey  // Use compressed key instead of original
    );

    // Add metadata to track this replacement
    newRotOp->setAttr("uses_compressed_key", BoolAttr::get(ctx, true));

    // Replace all uses of the original rotation with the new one
    originalRotOp.getResult().replaceAllUsesWith(newRotOp.getResult());

    // Remove the original rotation operation
    originalRotOp.erase();

    llvm::errs() << "Replaced rotation operation with compressed key version\n";
  }

  Value createCompressedKeyAfterOperation(openfhe::RotOp insertAfterOp,
                                          Value keyToCompress,
                                          unsigned targetLevel,
                                          int32_t rotationIndex,
                                          MLIRContext* ctx) {
    // This method is now deprecated - use createCompressedKeyAfterRotation
    // instead Keeping for bootstrap compression compatibility
    return createCompressedKeyAfterRotation(insertAfterOp, keyToCompress,
                                            targetLevel, rotationIndex, ctx);
  }

  Value insertCompressionAfterOperation(openfhe::RotOp rotOp,
                                        unsigned targetLevel,
                                        int32_t rotationIndex,
                                        MLIRContext* ctx) {
    // This method should only be used for bootstrap compressions now
    // For multi-level rotations, use insertCompressionWithDominancePreservation
    // instead

    OpBuilder builder(rotOp);
    builder.setInsertionPointAfter(rotOp);

    // Create compress_key operation right after the rotation
    auto compressOp = builder.create<openfhe::CompressKeyOp>(
        rotOp.getLoc(), rotOp.getEvalKey().getType(), rotOp.getCryptoContext(),
        rotOp.getEvalKey(),  // Compress the rotation's key
        builder.getI32IntegerAttr(targetLevel));

    // Add metadata for tracking
    compressOp->setAttr("bootstrap_compression", BoolAttr::get(ctx, true));
    compressOp->setAttr(
        "target_level",
        IntegerAttr::get(IntegerType::get(ctx, 32), targetLevel));
    compressOp->setAttr(
        "rotation_index",
        IntegerAttr::get(IntegerType::get(ctx, 32), rotationIndex));

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
