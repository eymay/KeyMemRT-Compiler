#include "lib/Transforms/AddRotationKeys/AddRotationKeys.h"

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Support/LLVM.h"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ADDROTATIONKEYS
#include "lib/Transforms/AddRotationKeys/AddRotationKeys.h.inc"

std::vector<int64_t> parseRotationIndices(const std::string& input) {
  std::vector<int64_t> indices;
  if (input.empty()) {
    return indices;
  }

  std::stringstream ss(input);
  std::string token;

  while (std::getline(ss, token, ',')) {
    // Trim whitespace
    token.erase(0, token.find_first_not_of(" \t"));
    token.erase(token.find_last_not_of(" \t") + 1);

    if (!token.empty()) {
      try {
        int64_t value = std::stoll(token);
        indices.push_back(value);
      } catch (const std::exception& e) {
        llvm::errs() << "Warning: Could not parse rotation index '" << token
                     << "': " << e.what() << "\n";
      }
    }
  }

  return indices;
}

// Convert DenseI64ArrayAttr to vector
std::vector<int64_t> denseArrayToVector(DenseI64ArrayAttr attr) {
  ArrayRef<int64_t> arrayRef = attr;
  return std::vector<int64_t>(arrayRef.begin(), arrayRef.end());
}

std::vector<int64_t> arrayRefToVector(ArrayRef<int64_t> arrayRef) {
  return std::vector<int64_t>(arrayRef.begin(), arrayRef.end());
}

struct AddRotationKeys : impl::AddRotationKeysBase<AddRotationKeys> {
  using impl::AddRotationKeysBase<AddRotationKeys>::AddRotationKeysBase;

  void runOnOperation() override {
    Operation* op = getOperation();
    MLIRContext* ctx = op->getContext();
    OpBuilder builder(ctx);

    // Parse the rotation indices from command line
    std::vector<int64_t> additionalIndices =
        parseRotationIndices(rotationIndices.getValue());

    if (additionalIndices.empty()) {
      llvm::errs()
          << "Warning: No rotation indices provided via --add-rotation-keys\n";
      return;
    }

    llvm::errs() << "Adding " << additionalIndices.size()
                 << " rotation indices: [";
    for (size_t i = 0; i < additionalIndices.size(); ++i) {
      if (i > 0) llvm::errs() << ", ";
      llvm::errs() << additionalIndices[i];
      if (i >= 9) {  // Limit output for readability
        llvm::errs() << ", ...";
        break;
      }
    }
    llvm::errs() << "]\n";

    // Convert to set for easier merging
    std::set<int64_t> additionalSet(additionalIndices.begin(),
                                    additionalIndices.end());

    // Process all GenRotKeyOp operations
    SmallVector<openfhe::GenRotKeyOp> genRotKeyOps;
    op->walk([&](openfhe::GenRotKeyOp genRotKeyOp) {
      genRotKeyOps.push_back(genRotKeyOp);
      return WalkResult::advance();
    });

    for (auto genRotKeyOp : genRotKeyOps) {
      // Get current indices
      std::vector<int64_t> currentIndices;
      auto indicesAttr = genRotKeyOp.getIndices();
      if (!indicesAttr.empty()) {
        currentIndices = arrayRefToVector(indicesAttr);
      }

      // Convert current indices to set
      std::set<int64_t> currentSet(currentIndices.begin(),
                                   currentIndices.end());

      // Check for keys that are not in the additional list
      std::set<int64_t> notInAdditional;
      std::set_difference(
          currentSet.begin(), currentSet.end(), additionalSet.begin(),
          additionalSet.end(),
          std::inserter(notInAdditional, notInAdditional.begin()));

      if (!notInAdditional.empty()) {
        llvm::errs() << "Warning: Found existing rotation keys not in the "
                        "provided list: [";
        bool first = true;
        for (int64_t idx : notInAdditional) {
          if (!first) llvm::errs() << ", ";
          llvm::errs() << idx;
          first = false;
        }
        llvm::errs() << "]\n";
      }

      // Determine final indices based on overwrite option
      std::vector<int64_t> finalIndices;
      std::set<int64_t> trulyNewIndices;

      if (overwrite) {
        // Overwrite mode: just use the additional indices
        finalIndices = additionalIndices;
        llvm::errs() << "Overwrite mode: replacing " << currentIndices.size()
                     << " existing keys with " << finalIndices.size()
                     << " new keys\n";
      } else {
        // Merge mode: find truly new indices and merge sets
        std::set_difference(
            additionalSet.begin(), additionalSet.end(), currentSet.begin(),
            currentSet.end(),
            std::inserter(trulyNewIndices, trulyNewIndices.begin()));

        // Merge the sets (union)
        std::set<int64_t> mergedSet;
        std::set_union(currentSet.begin(), currentSet.end(),
                       additionalSet.begin(), additionalSet.end(),
                       std::inserter(mergedSet, mergedSet.begin()));

        // Convert back to vector
        finalIndices = std::vector<int64_t>(mergedSet.begin(), mergedSet.end());

        llvm::errs() << "Merge mode: " << currentIndices.size()
                     << " existing + " << trulyNewIndices.size() << " new + "
                     << (additionalIndices.size() - trulyNewIndices.size())
                     << " already present = " << finalIndices.size()
                     << " total rotation keys\n";
      }

      // Create new DenseI64ArrayAttr
      auto newIndicesAttr = builder.getDenseI64ArrayAttr(finalIndices);

      // Update the operation
      genRotKeyOp.setIndicesAttr(newIndicesAttr);
    }

    // Also process GenRotKeyDepthOp operations if they exist
    SmallVector<openfhe::GenRotKeyDepthOp> genRotKeyDepthOps;
    op->walk([&](openfhe::GenRotKeyDepthOp genRotKeyDepthOp) {
      genRotKeyDepthOps.push_back(genRotKeyDepthOp);
      return WalkResult::advance();
    });

    for (auto genRotKeyDepthOp : genRotKeyDepthOps) {
      // Get current indices
      std::vector<int64_t> currentIndices;
      if (auto indicesAttr = genRotKeyDepthOp.getIndices()) {
        if (auto denseAttr = dyn_cast<DenseI64ArrayAttr>(indicesAttr)) {
          currentIndices = denseArrayToVector(denseAttr);
        } else if (auto arrayAttr = dyn_cast<ArrayAttr>(indicesAttr)) {
          for (Attribute attr : arrayAttr) {
            if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
              currentIndices.push_back(intAttr.getInt());
            }
          }
        }
      }

      // Convert current indices to set
      std::set<int64_t> currentSet(currentIndices.begin(),
                                   currentIndices.end());

      // Check for keys that are not in the additional list
      std::set<int64_t> notInAdditional;
      std::set_difference(
          currentSet.begin(), currentSet.end(), additionalSet.begin(),
          additionalSet.end(),
          std::inserter(notInAdditional, notInAdditional.begin()));

      if (!notInAdditional.empty()) {
        llvm::errs() << "Warning: Found existing rotation keys in "
                        "GenRotKeyDepthOp not in the provided list: [";
        bool first = true;
        for (int64_t idx : notInAdditional) {
          if (!first) llvm::errs() << ", ";
          llvm::errs() << idx;
          first = false;
        }
        llvm::errs() << "]\n";
      }

      // Determine final indices based on overwrite option
      std::vector<int64_t> finalIndices;
      std::set<int64_t> trulyNewIndices;

      if (overwrite) {
        // Overwrite mode: just use the additional indices
        finalIndices = additionalIndices;
        llvm::errs() << "Overwrite mode: replacing " << currentIndices.size()
                     << " existing keys with " << finalIndices.size()
                     << " new keys (GenRotKeyDepthOp)\n";
      } else {
        // Merge mode: find truly new indices and merge sets
        std::set_difference(
            additionalSet.begin(), additionalSet.end(), currentSet.begin(),
            currentSet.end(),
            std::inserter(trulyNewIndices, trulyNewIndices.begin()));

        // Merge the sets (union)
        std::set<int64_t> mergedSet;
        std::set_union(currentSet.begin(), currentSet.end(),
                       additionalSet.begin(), additionalSet.end(),
                       std::inserter(mergedSet, mergedSet.begin()));

        // Convert back to vector
        finalIndices = std::vector<int64_t>(mergedSet.begin(), mergedSet.end());

        llvm::errs() << "Merge mode: " << currentIndices.size()
                     << " existing + " << trulyNewIndices.size() << " new + "
                     << (additionalIndices.size() - trulyNewIndices.size())
                     << " already present = " << finalIndices.size()
                     << " total rotation keys (GenRotKeyDepthOp)\n";
      }

      // Create new ArrayAttr for GenRotKeyDepthOp (it uses ArrayAttr, not
      // DenseI64ArrayAttr)
      SmallVector<Attribute> indexAttrs;
      for (int64_t idx : finalIndices) {
        indexAttrs.push_back(builder.getI64IntegerAttr(idx));
      }
      auto newIndicesAttr = builder.getArrayAttr(indexAttrs);

      // Update the operation
      genRotKeyDepthOp.setIndicesAttr(newIndicesAttr);
    }

    if (genRotKeyOps.empty() && genRotKeyDepthOps.empty()) {
      llvm::errs() << "Warning: No GenRotKeyOp or GenRotKeyDepthOp operations "
                      "found to update\n";
    }
  }
};

}  // namespace heir
}  // namespace mlir
