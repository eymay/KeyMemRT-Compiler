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

// Simple profile entry from your LOG_CT output
struct ProfileEntry {
  std::string inputMLIRName;  // e.g., "%ct_188"
  unsigned level;             // e.g., 0
  unsigned noise;
  unsigned towers;
  unsigned line;
};

// Parse the profile data from your format
class SimpleProfileParser {
 public:
  static std::vector<ProfileEntry> parseProfile(const std::string& filename) {
    std::vector<ProfileEntry> entries;
    std::ifstream file(filename);
    std::string line;

    if (!file.good()) {
      return entries;
    }

    while (std::getline(file, line)) {
      // Parse:
      // "ROTATION_PROFILE::INPUT:%ct_188:LEVEL:0:NOISE:2:TOWERS:31:LINE:1077"
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
      // Format:
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

struct KeyCompression : impl::KeyCompressionBase<KeyCompression> {
  void runOnOperation() override {
    Operation* op = getOperation();
    MLIRContext* ctx = op->getContext();

    // Step 1: Load profile data
    auto profileEntries =
        SimpleProfileParser::parseProfile("rotation_profile.txt");
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

    // Step 3: Use dataflow analysis to determine which deserialize ops belong
    // to which bootstrap
    std::map<int32_t, unsigned> rotationToOptimalLevel;

    // Build a map of all deserialize operations by rotation index
    std::map<int32_t, std::vector<openfhe::DeserializeKeyOp>> allDeserializeOps;
    op->walk([&](openfhe::DeserializeKeyOp deserOp) {
      if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
        int32_t rotationIndex = indexAttr.getInt();
        allDeserializeOps[rotationIndex].push_back(deserOp);
      }
    });

    // Process each bootstrap and find which specific deserialize ops it uses
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
          llvm::errs() << "Bootstrap input " << bootstrapInputMLIRName
                       << " is at level " << bootstrapInputLevel << "\n";
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
        llvm::errs() << "Bootstrap uses " << bootstrapRotationIndices.size()
                     << " rotation indices at level " << bootstrapInputLevel
                     << "\n";
      }

      // For each rotation index used by this bootstrap, find which deserialize
      // op is live
      for (int32_t rotationIndex : bootstrapRotationIndices) {
        auto it = allDeserializeOps.find(rotationIndex);
        if (it == allDeserializeOps.end()) continue;

        // Find the deserialize operation that is live at this bootstrap
        openfhe::DeserializeKeyOp liveDeserOp = nullptr;
        for (auto deserOp : it->second) {
          if (isKeyLiveAtBootstrap(deserOp, bootstrapOp)) {
            liveDeserOp = deserOp;
            break;
          }
        }

        if (liveDeserOp) {
          // Set level for this specific deserialize operation, not the rotation
          // index globally
          setLevelForSpecificDeserializeOp(liveDeserOp, bootstrapInputLevel,
                                           rotationIndex);
          llvm::errs() << "Set level " << bootstrapInputLevel
                       << " for deserialize of rotation " << rotationIndex
                       << " (bootstrap input: " << bootstrapInputMLIRName
                       << ")\n";
        } else {
          llvm::errs() << "Warning: No live deserialize found for rotation "
                       << rotationIndex << " at bootstrap with input "
                       << bootstrapInputMLIRName << "\n";
        }
      }
    });

    // Step 4: Handle regular rotation operations from profile data
    for (const auto& entry : profileEntries) {
      auto it = mlirInputToRotIndex.find(entry.inputMLIRName);
      if (it != mlirInputToRotIndex.end()) {
        int32_t rotationIndex = it->second;

        // Take minimum if this rotation is already set by bootstrap
        if (rotationToOptimalLevel.find(rotationIndex) !=
            rotationToOptimalLevel.end()) {
          unsigned existingLevel = rotationToOptimalLevel[rotationIndex];
          unsigned newLevel = std::min(existingLevel, entry.level);
          rotationToOptimalLevel[rotationIndex] = newLevel;
          llvm::errs() << "Regular rotation " << rotationIndex
                       << ": updated level from " << existingLevel << " to "
                       << newLevel << "\n";
        } else {
          rotationToOptimalLevel[rotationIndex] = entry.level;
          llvm::errs() << "Regular rotation " << rotationIndex
                       << ": set level to " << entry.level << "\n";
        }
      }
    }

    // Step 5: Set key_depth attributes based on our analysis (not global
    // rotation index mapping)
    op->walk([&](openfhe::DeserializeKeyOp deserOp) {
      // Check if this specific deserialize operation has been assigned a level
      if (deserOp->hasAttr("assigned_key_depth")) {
        // Already processed by bootstrap analysis
        return;
      }

      // Handle regular rotations (non-bootstrap)
      int32_t rotationIndex = -1;
      if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
        rotationIndex = indexAttr.getInt();
      }

      if (rotationIndex == -1) {
        llvm::errs()
            << "Warning: Could not get rotation index from deserialize op\n";
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
      } else {
        llvm::errs() << "No profile data for rotation " << rotationIndex
                     << " - keeping default\n";
      }
    });

    // Step 6: Generate GenRotKeyDepth operations based on deserialize levels
    generateGenRotKeyDepthOps(op, ctx);

    // Step 7: Enhanced report
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

    llvm::errs() << "\nDeserialize operations by level:\n";
    for (const auto& [level, count] : levelCounts) {
      llvm::errs() << "  Level " << level << ": " << count << " keys\n";
    }
    llvm::errs() << "=== End Report ===\n";
  }

 private:
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
      auto genRotKeyDepthOp = builder.create<openfhe::GenRotKeyDepthOp>(
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

  // Check if a deserialize operation's key is live at a specific bootstrap
  // operation
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

  // Set level for a specific deserialize operation
  void setLevelForSpecificDeserializeOp(openfhe::DeserializeKeyOp deserOp,
                                        unsigned level, int32_t rotationIndex) {
    MLIRContext* ctx = deserOp.getContext();
    deserOp->setAttr("key_depth",
                     IntegerAttr::get(IntegerType::get(ctx, 64), level));
    deserOp->setAttr("assigned_key_depth",
                     BoolAttr::get(ctx, true));  // Mark as processed
  }

  // Check if op1 comes before op2 in program order
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

  // Check if a key is cleared between two operations
  bool isKeyClearedBetween(Value evalKey, Operation* startOp,
                           Operation* endOp) {
    Operation* current = startOp->getNextNode();

    while (current != nullptr && current != endOp) {
      if (auto clearOp = dyn_cast<openfhe::ClearKeyOp>(current)) {
        if (clearOp.getEvalKey() == evalKey) {
          llvm::errs() << "Key is cleared between deserialize and bootstrap\n";
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
