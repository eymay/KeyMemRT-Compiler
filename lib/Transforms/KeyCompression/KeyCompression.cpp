#include "lib/Transforms/KeyCompression/KeyCompression.h"

#include <algorithm>
#include <map>
#include <set>
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

struct KeyCompression : impl::KeyCompressionBase<KeyCompression> {
  using impl::KeyCompressionBase<KeyCompression>::KeyCompressionBase;

  void runOnOperation() override {
    Operation* op = getOperation();
    MLIRContext* ctx = op->getContext();

    llvm::errs()
        << "Starting key compression pass using key_level attributes\n";

    // Step 1: Look for DeserializeKeyOp operations with key_level attributes
    // and convert them to key_depth attributes (key_depth = key_level for our
    // purposes)
    unsigned convertedKeys = 0;

    op->walk([&](openfhe::DeserializeKeyOp deserOp) {
      if (auto levelAttr = deserOp->getAttrOfType<IntegerAttr>("key_level")) {
        unsigned level = levelAttr.getInt();

        // Set the key_depth attribute to match the level
        // (key_depth and key_level are the same concept - depth of compression)
        deserOp->setAttr("key_depth",
                         IntegerAttr::get(IntegerType::get(ctx, 64), level));

        // llvm::errs() << "Set key_depth=" << level << " for DeserializeKeyOp "
        //              << "(from key_level=" << level << ")\n";
        convertedKeys++;
      }
    });

    if (convertedKeys == 0) {
      llvm::errs()
          << "No DeserializeKeyOp operations with key_level attributes found. "
          << "Make sure the profile-annotator pass has been run first.\n";
      return;
    }

    llvm::errs() << "Converted " << convertedKeys << " key operations\n";

    // Step 2: Generate GenRotKeyDepth operations based on key_depth attributes
    generateGenRotKeyDepthOps(op, ctx);

    // Step 3: Generate report
    generateReport(op);
  }

 private:
  void generateGenRotKeyDepthOps(Operation* op, MLIRContext* ctx) {
    // Find existing GenRotKeyOp
    openfhe::GenRotKeyOp existingGenRotKeyOp;
    op->walk([&](openfhe::GenRotKeyOp genRotKeyOp) {
      existingGenRotKeyOp = genRotKeyOp;
      return WalkResult::interrupt();
    });

    if (!existingGenRotKeyOp) {
      llvm::errs() << "Warning: No existing GenRotKeyOp found to replace\n";
      return;
    }

    // Collect rotation indices by depth from deserialize operations
    std::map<unsigned, std::vector<int32_t>> depthToRotationIndices;

    op->walk([&](openfhe::DeserializeKeyOp deserOp) {
      if (auto keyDepthAttr =
              deserOp->getAttrOfType<IntegerAttr>("key_depth")) {
        unsigned depth = keyDepthAttr.getInt();

        // Find rotation operations that use this deserialized key
        for (auto user : deserOp.getResult().getUsers()) {
          if (auto rotOp = dyn_cast<openfhe::RotOp>(user)) {
            auto rotationIndex =
                rotOp.getEvalKey().getType().getRotationIndex().getInt();
            depthToRotationIndices[depth].push_back(rotationIndex);

            // llvm::errs() << "Found rotation " << rotationIndex << " at depth
            // "
            //              << depth << "\n";
          }
        }
      }
    });

    if (depthToRotationIndices.empty()) {
      llvm::errs() << "Warning: No rotation operations found that use keyed "
                      "operations with key_depth\n";
      return;
    }

    // Create GenRotKeyDepthOp for each depth level
    OpBuilder builder(existingGenRotKeyOp);

    for (const auto& [depth, rotationIndices] : depthToRotationIndices) {
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
          IntegerAttr::get(IntegerType::get(ctx, 64), depth));

      llvm::errs() << "Created GenRotKeyDepthOp for depth " << depth << " with "
                   << sortedIndices.size() << " rotation indices: ";
      for (auto idx : sortedIndices) {
        llvm::errs() << idx << " ";
      }
      llvm::errs() << "\n";
    }

    // Remove the original GenRotKeyOp
    existingGenRotKeyOp.erase();
    llvm::errs() << "Removed original GenRotKeyOp\n";
  }

  void generateReport(Operation* op) {
    llvm::errs() << "\n=== Key Compression Report ===\n";

    // Count deserialize operations by depth
    std::map<unsigned, int> depthCounts;
    std::map<unsigned, std::vector<int32_t>> depthToRotations;

    op->walk([&](openfhe::DeserializeKeyOp deserOp) {
      if (auto keyDepthAttr =
              deserOp->getAttrOfType<IntegerAttr>("key_depth")) {
        unsigned depth = keyDepthAttr.getInt();
        depthCounts[depth]++;

        // Find which rotations use this key
        for (auto user : deserOp.getResult().getUsers()) {
          if (auto rotOp = dyn_cast<openfhe::RotOp>(user)) {
            auto rotationIndex =
                rotOp.getEvalKey().getType().getRotationIndex().getInt();
            depthToRotations[depth].push_back(rotationIndex);
          }
        }
      }
    });

    // Count operations with key_level attributes (before conversion)
    int keyLevelCount = 0;
    op->walk([&](openfhe::DeserializeKeyOp deserOp) {
      if (deserOp->hasAttr("key_level")) {
        keyLevelCount++;
      }
    });

    llvm::errs() << "DeserializeKeyOp operations with key_level: "
                 << keyLevelCount << "\n";
    llvm::errs() << "\nKey operations by depth level:\n";
    for (const auto& [depth, count] : depthCounts) {
      llvm::errs() << "  Depth " << depth << ": " << count << " keys";
      if (!depthToRotations[depth].empty()) {
        llvm::errs() << " (rotations: ";
        for (size_t i = 0; i < depthToRotations[depth].size(); ++i) {
          if (i > 0) llvm::errs() << ", ";
          llvm::errs() << depthToRotations[depth][i];
          if (i >= 4) {  // Limit output for readability
            llvm::errs() << "...";
            break;
          }
        }
        llvm::errs() << ")";
      }
      llvm::errs() << "\n";
    }

    // Count total rotation operations
    int totalRotations = 0;
    op->walk([&](openfhe::RotOp rotOp) { totalRotations++; });

    // Count GenRotKeyDepthOp operations
    int genRotKeyDepthOps = 0;
    op->walk([&](openfhe::GenRotKeyDepthOp genRotKeyDepthOp) {
      genRotKeyDepthOps++;
    });

    llvm::errs() << "\nTotal rotation operations: " << totalRotations << "\n";
    llvm::errs() << "GenRotKeyDepthOp operations created: " << genRotKeyDepthOps
                 << "\n";
    llvm::errs() << "=== End Key Compression Report ===\n\n";
  }
};

}  // namespace

}  // namespace heir
}  // namespace mlir
