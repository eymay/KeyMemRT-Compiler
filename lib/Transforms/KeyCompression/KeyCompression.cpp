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

    // Step 1: Identify bootstrap rotation indices that should not be compressed
    std::set<int32_t> bootstrapRotationIndices =
        getBootstrapRotationIndices(op);

    if (!bootstrapRotationIndices.empty()) {
      llvm::errs() << "Found " << bootstrapRotationIndices.size()
                   << " bootstrap rotation indices that will be protected from "
                      "compression: ";
      for (auto idx : bootstrapRotationIndices) {
        llvm::errs() << idx << " ";
      }
      llvm::errs() << "\n";
    }

    // Step 2: Look for DeserializeKeyOp operations with key_level attributes
    // and convert them to key_depth attributes, but skip bootstrap keys
    unsigned convertedKeys = 0;
    unsigned protectedBootstrapKeys = 0;

    op->walk([&](openfhe::DeserializeKeyOp deserOp) {
      if (auto levelAttr = deserOp->getAttrOfType<IntegerAttr>("key_level")) {
        unsigned level = levelAttr.getInt();

        // Check if this key is used for bootstrap operations
        bool isBootstrapKey =
            isKeyUsedForBootstrap(deserOp, bootstrapRotationIndices);

        if (isBootstrapKey) {
          // For bootstrap keys, do NOT set key_depth to preserve them at their
          // original level
          // llvm::errs() << "Protected bootstrap key from compression
          // (key_level="
          //              << level << ")\n";
          protectedBootstrapKeys++;
          return;
        }

        // Set the key_depth attribute to match the level for non-bootstrap keys
        // (key_depth and key_level are the same concept - depth of compression)
        deserOp->setAttr("key_depth",
                         IntegerAttr::get(IntegerType::get(ctx, 64), level));

        convertedKeys++;
      }
    });

    if (convertedKeys == 0 && protectedBootstrapKeys == 0) {
      llvm::errs()
          << "No DeserializeKeyOp operations with key_level attributes found. "
          << "Make sure the profile-annotator pass has been run first.\n";
      return;
    }

    llvm::errs() << "Converted " << convertedKeys << " key operations\n";
    llvm::errs() << "Protected " << protectedBootstrapKeys
                 << " bootstrap keys from compression\n";

    // Step 3: Generate GenRotKeyDepth operations based on key_depth attributes
    generateGenRotKeyDepthOps(op, ctx);

    // Step 4: Generate report
    generateReport(op);
  }

 private:
  // Extract bootstrap rotation indices from bootstrap operations
  std::set<int32_t> getBootstrapRotationIndices(Operation* op) {
    std::set<int32_t> bootstrapIndices;

    op->walk([&](openfhe::BootstrapOp bootstrapOp) {
      // Check if the bootstrap operation has rotation_indices attribute
      if (auto rotationIndicesAttr =
              bootstrapOp->getAttrOfType<ArrayAttr>("rotation_indices")) {
        for (auto indexAttr : rotationIndicesAttr.getAsRange<IntegerAttr>()) {
          int32_t rotationIndex = indexAttr.getInt();
          bootstrapIndices.insert(rotationIndex);
        }
      }
    });

    return bootstrapIndices;
  }

  // Check if a deserialize key operation is used for bootstrap
  bool isKeyUsedForBootstrap(
      openfhe::DeserializeKeyOp deserOp,
      const std::set<int32_t>& bootstrapRotationIndices) {
    // Get the rotation index for this deserialize operation
    if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
      int32_t rotationIndex = indexAttr.getInt();
      return bootstrapRotationIndices.count(rotationIndex) > 0;
    }

    // Alternative method: check if the key is used by bootstrap operations
    // Look at users of this deserialize operation
    Value evalKey = deserOp.getResult();
    for (auto user : evalKey.getUsers()) {
      if (auto rotOp = dyn_cast<openfhe::RotOp>(user)) {
        // Check if this rotation is within a bootstrap's lifetime
        if (isRotationWithinBootstrapLifetime(rotOp,
                                              bootstrapRotationIndices)) {
          return true;
        }
      }
    }

    return false;
  }

  // Check if a rotation operation occurs within a bootstrap's key lifetime
  bool isRotationWithinBootstrapLifetime(
      openfhe::RotOp rotOp, const std::set<int32_t>& bootstrapRotationIndices) {
    auto rotationIndex =
        rotOp.getEvalKey().getType().getRotationIndex();

    // If this rotation index is used by bootstrap, check if it's within
    // bootstrap lifetime
    if (bootstrapRotationIndices.count(rotationIndex) > 0) {
      // Look for nearby bootstrap operations that might use this rotation
      Operation* rotOperation = rotOp.getOperation();

      // Search backwards and forwards for bootstrap operations
      Operation* current = rotOperation;
      while (current != nullptr) {
        if (auto bootstrapOp = dyn_cast<openfhe::BootstrapOp>(current)) {
          // Check if this bootstrap uses our rotation index
          if (auto rotationIndicesAttr =
                  bootstrapOp->getAttrOfType<ArrayAttr>("rotation_indices")) {
            for (auto indexAttr :
                 rotationIndicesAttr.getAsRange<IntegerAttr>()) {
              if (indexAttr.getInt() == rotationIndex) {
                return true;
              }
            }
          }
        }
        current = current->getPrevNode();
        if (current == nullptr) break;
      }

      // Search forward
      current = rotOperation->getNextNode();
      while (current != nullptr) {
        if (auto bootstrapOp = dyn_cast<openfhe::BootstrapOp>(current)) {
          if (auto rotationIndicesAttr =
                  bootstrapOp->getAttrOfType<ArrayAttr>("rotation_indices")) {
            for (auto indexAttr :
                 rotationIndicesAttr.getAsRange<IntegerAttr>()) {
              if (indexAttr.getInt() == rotationIndex) {
                return true;
              }
            }
          }
        }
        current = current->getNextNode();
        if (current == nullptr) break;
      }
    }

    return false;
  }

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
                rotOp.getEvalKey().getType().getRotationIndex();
            depthToRotationIndices[depth].push_back(rotationIndex);
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
                rotOp.getEvalKey().getType().getRotationIndex();
            depthToRotations[depth].push_back(rotationIndex);
          }
        }
      }
    });

    // Count operations with key_level attributes (before conversion)
    int keyLevelCount = 0;
    int protectedBootstrapCount = 0;
    op->walk([&](openfhe::DeserializeKeyOp deserOp) {
      if (deserOp->hasAttr("key_level")) {
        keyLevelCount++;
        // If it has key_level but no key_depth, it was protected as bootstrap
        // key
        if (!deserOp->hasAttr("key_depth")) {
          protectedBootstrapCount++;
        }
      }
    });

    llvm::errs() << "DeserializeKeyOp operations with key_level: "
                 << keyLevelCount << "\n";
    llvm::errs() << "Protected bootstrap keys: " << protectedBootstrapCount
                 << "\n";
    // llvm::errs() << "\nKey operations by depth level:\n";
    // for (const auto& [depth, count] : depthCounts) {
    //   // llvm::errs() << "  Depth " << depth << ": " << count << " keys";
    //   if (!depthToRotations[depth].empty()) {
    //     llvm::errs() << " (rotations: ";
    //     for (size_t i = 0; i < depthToRotations[depth].size(); ++i) {
    //       if (i > 0) llvm::errs() << ", ";
    //       llvm::errs() << depthToRotations[depth][i];
    //       if (i >= 4) {  // Limit output for readability
    //         llvm::errs() << "...";
    //         break;
    //       }
    //     }
    //     llvm::errs() << ")";
    //   }
    //   llvm::errs() << "\n";
    // }

    // Count total rotation operations
    int totalRotations = 0;
    op->walk([&](openfhe::RotOp rotOp) { totalRotations++; });

    // Count GenRotKeyDepthOp operations
    int genRotKeyDepthOps = 0;
    op->walk([&](openfhe::GenRotKeyDepthOp genRotKeyDepthOp) {
      genRotKeyDepthOps++;
    });

    // Count bootstrap operations
    int bootstrapOps = 0;
    op->walk([&](openfhe::BootstrapOp bootstrapOp) { bootstrapOps++; });

    llvm::errs() << "\nTotal rotation operations: " << totalRotations << "\n";
    llvm::errs() << "Bootstrap operations: " << bootstrapOps << "\n";
    llvm::errs() << "GenRotKeyDepthOp operations created: " << genRotKeyDepthOps
                 << "\n";
    llvm::errs() << "=== End Key Compression Report ===\n\n";
  }
};

}  // namespace

}  // namespace heir
}  // namespace mlir
