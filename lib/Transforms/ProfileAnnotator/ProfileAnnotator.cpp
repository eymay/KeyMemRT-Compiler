#include "lib/Transforms/ProfileAnnotator/ProfileAnnotator.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

#include "lib/Dialect/KMRT/IR/KMRTOps.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Support/LLVM.h"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_PROFILEANNOTATOR
#include "lib/Transforms/ProfileAnnotator/ProfileAnnotator.h.inc"

namespace {

// Profile data structure
struct ProfileEntry {
  std::string mlirName;
  bool isInput =
      false;  // true for ROTATION_PROFILE::INPUT, false for PROFILE::OUTPUT
  unsigned level = 0;
  unsigned noise = 0;
  unsigned towers = 0;
  unsigned line = 0;
};

// Parser for profile data
class ProfileParser {
 public:
  static std::vector<ProfileEntry> parseProfile(const std::string& filename) {
    std::vector<ProfileEntry> entries;
    std::ifstream file(filename);
    std::string line;

    if (!file.good()) {
      llvm::errs() << "Warning: Could not open profile file: " << filename
                   << "\n";
      return entries;
    }

    while (std::getline(file, line)) {
      // Parse format from LOG_CT macro:
      // "PROFILE::OUTPUT:mlir_name:LEVEL:level:NOISE:noise:TOWERS:towers:LINE:line"
      if (line.find("PROFILE:") == 0 || line.find("ROTATION_PROFILE:") == 0) {
        auto entry = parseLine(line);
        if (!entry.mlirName.empty()) {
          entries.push_back(entry);
        }
      }
    }

    llvm::errs() << "Loaded " << entries.size() << " profile entries from "
                 << filename << "\n";
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
      // PROFILE::OUTPUT:mlir_name:LEVEL:level:NOISE:noise:TOWERS:towers:LINE:line
      if (parts[0] == "PROFILE" && parts.size() >= 10 && parts[1].empty() &&
          parts[2] == "OUTPUT") {
        // PROFILE::OUTPUT:mlir_name:LEVEL:level:NOISE:noise:TOWERS:towers:LINE:line
        entry.mlirName = parts[3];
        entry.isInput = false;
        entry.level = std::stoul(parts[5]);
        entry.noise = std::stoul(parts[7]);
        entry.towers = std::stoul(parts[9]);
        if (parts.size() >= 12) {
          entry.line = std::stoul(parts[11]);
        }
      } else if (parts[0] == "ROTATION_PROFILE" && parts.size() >= 10 &&
                 parts[1].empty() && parts[2] == "INPUT") {
        // ROTATION_PROFILE::INPUT:mlir_name:LEVEL:level:NOISE:noise:TOWERS:towers:LINE:line
        entry.mlirName = parts[3];
        entry.isInput = true;
        entry.level = std::stoul(parts[5]);
        entry.noise = std::stoul(parts[7]);
        entry.towers = std::stoul(parts[9]);
        if (parts.size() >= 12) {
          entry.line = std::stoul(parts[11]);
        }
      }
    } catch (const std::exception& e) {
      llvm::errs() << "Warning: Failed to parse profile line: " << line << " ("
                   << e.what() << ")\n";
      entry.mlirName.clear();  // Mark as invalid
    }

    return entry;
  }
};

struct ProfileAnnotator : impl::ProfileAnnotatorBase<ProfileAnnotator> {
  using impl::ProfileAnnotatorBase<ProfileAnnotator>::ProfileAnnotatorBase;

  void runOnOperation() override {
    Operation* op = getOperation();
    MLIRContext* ctx = op->getContext();

    // Step 1: Load profile data
    auto profileEntries = ProfileParser::parseProfile(profileFile.getValue());
    if (profileEntries.empty()) {
      llvm::errs() << "No profile data found. Skipping annotation.\n";
      signalPassFailure();
      return;
    }

    // Step 2: Build mapping from MLIR result names to profile entries
    // Group by name and sort by line number to handle duplicate names
    std::map<std::string, std::vector<ProfileEntry>> outputEntriesByName;
    std::map<std::string, std::vector<ProfileEntry>> inputEntriesByName;

    for (const auto& entry : profileEntries) {
      if (entry.isInput) {
        inputEntriesByName[entry.mlirName].push_back(entry);
      } else {
        outputEntriesByName[entry.mlirName].push_back(entry);
      }
    }

    // Sort each vector by line number for consistent ordering
    for (auto& [name, entries] : outputEntriesByName) {
      std::sort(entries.begin(), entries.end(),
                [](const ProfileEntry& a, const ProfileEntry& b) {
                  return a.line < b.line;
                });
    }
    for (auto& [name, entries] : inputEntriesByName) {
      std::sort(entries.begin(), entries.end(),
                [](const ProfileEntry& a, const ProfileEntry& b) {
                  return a.line < b.line;
                });
    }

    // Track which occurrence (by line order) we're at for each name
    std::map<std::string, unsigned> outputOccurrenceCount;
    std::map<std::string, unsigned> inputOccurrenceCount;

    // Step 3: Walk all operations and annotate those with matching results
    unsigned annotatedOps = 0;
    op->walk([&](Operation* operation) {
      // Process each result of the operation
      for (auto result : operation->getResults()) {
        std::string resultName = getMLIRNameForValue(result);

        auto it = outputEntriesByName.find(resultName);
        if (it != outputEntriesByName.end()) {
          const auto& entries = it->second;
          unsigned& occurrenceIdx = outputOccurrenceCount[resultName];

          // Use the entry corresponding to this occurrence (by line order)
          if (occurrenceIdx < entries.size()) {
            const auto& entry = entries[occurrenceIdx];
            occurrenceIdx++;

            // Annotate the operation with the towers count
            operation->setAttr(
                "result_towers",
                IntegerAttr::get(IntegerType::get(ctx, 32), entry.towers));

            // llvm::errs() << "Annotated operation with result " << resultName
            //              << " (occurrence " << (occurrenceIdx - 1)
            //              << ", line " << entry.line << ") with "
            //              << entry.towers << " towers\n";
            annotatedOps++;

            // Since we found a match for this operation, we can break
            // (one annotation per operation is sufficient)
            break;
          }
        }
      }
    });

    // Step 4: Walk rotation operations and annotate LoadKeyOp with key_level
    auto annotateRotationKey = [&](Value ciphertext, Value evalKey) {
      std::string ctName = getMLIRNameForValue(ciphertext);

      // Check if this ciphertext is in our input profile data
      auto it = inputEntriesByName.find(ctName);
      if (it != inputEntriesByName.end()) {
        const auto& entries = it->second;
        unsigned& occurrenceIdx = inputOccurrenceCount[ctName];

        // Use the entry corresponding to this occurrence (by line order)
        if (occurrenceIdx < entries.size()) {
          const auto& entry = entries[occurrenceIdx];
          occurrenceIdx++;

          // Found a rotation with profiled input - now find the key source
          // Trace back to find the LoadKeyOp that produced this key
          if (auto defOp = evalKey.getDefiningOp()) {
            if (auto loadOp = dyn_cast<kmrt::LoadKeyOp>(defOp)) {
              // Annotate this LoadKeyOp with level info (using key_level)
              loadOp->setAttr(
                  "key_level",
                  IntegerAttr::get(IntegerType::get(ctx, 32), entry.level));
              // llvm::errs() << "Annotated LoadKeyOp for rotation of " <<
              // ctName
              //              << " (occurrence " << (occurrenceIdx - 1)
              //              << ", line " << entry.line << ") with key_level="
              //              << entry.level << "\n";
            }
          }
        }
      }
    };

    op->walk([&](openfhe::RotOp rotOp) {
      annotateRotationKey(rotOp.getCiphertext(), rotOp.getEvalKey());
    });

    op->walk([&](openfhe::FastRotationOp fastRotOp) {
      annotateRotationKey(fastRotOp.getCiphertext(), fastRotOp.getEvalKey());
    });

    llvm::errs() << "Profile annotation complete: " << annotatedOps
                 << " operations annotated with result_towers\n";
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
};

}  // namespace
}  // namespace heir
}  // namespace mlir
