#include "lib/Transforms/ProfileAnnotator/ProfileAnnotator.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

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
  std::string outputMLIRName;
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
      if (line.find("PROFILE:") == 0) {
        auto entry = parseLine(line);
        if (!entry.outputMLIRName.empty()) {
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
      if (parts.size() >= 10 &&
          parts[1].empty()) {  // parts[1] is empty because of "::"
        entry.outputMLIRName = parts[3];
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
      entry.outputMLIRName.clear();  // Mark as invalid
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
      return;
    }

    // Step 2: Build mapping from MLIR result names to profile entries
    std::map<std::string, unsigned> mlirNameToTowers;
    for (const auto& entry : profileEntries) {
      mlirNameToTowers[entry.outputMLIRName] = entry.towers;
      llvm::errs() << "Profile: " << entry.outputMLIRName << " -> "
                   << entry.towers << " towers\n";
    }

    // Step 3: Walk all operations and annotate those with matching results
    unsigned annotatedOps = 0;
    op->walk([&](Operation* operation) {
      // Process each result of the operation
      for (auto result : operation->getResults()) {
        std::string resultName = getMLIRNameForValue(result);

        auto it = mlirNameToTowers.find(resultName);
        if (it != mlirNameToTowers.end()) {
          // Found a match - annotate the operation with the towers count
          operation->setAttr(
              "result_towers",
              IntegerAttr::get(IntegerType::get(ctx, 32), it->second));

          llvm::errs() << "Annotated operation with result " << resultName
                       << " with " << it->second << " towers\n";
          annotatedOps++;

          // Since we found a match for this operation, we can break
          // (one annotation per operation is sufficient)
          break;
        }
      }
    });

    llvm::errs() << "Profile annotation complete: " << annotatedOps
                 << " operations annotated\n";
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
