#include "lib/Transforms/KeyCompression/KeyCompression.h"

#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
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
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *ctx = op->getContext();

    // Maps each Value to its current multiplicative depth
    llvm::DenseMap<Value, unsigned> depthMap;

    // Maps each deserialize op to its minimum required depth
    llvm::DenseMap<Operation *, unsigned> deserializeDepthMap;

    // Maps each rotation index to the depths it's used at
    llvm::DenseMap<int64_t, std::set<unsigned>> indexToDepths;

    // Initialize depth for block arguments
    op->walk([&](Block *block) {
      for (BlockArgument arg : block->getArguments()) {
        depthMap[arg] = 0;
      }
    });

    // First pass: Calculate depths for all values
    op->walk([&](Operation *operation) {
      // Calculate max depth of operands
      unsigned maxDepth = 0;
      for (Value operand : operation->getOperands()) {
        if (depthMap.count(operand) > 0) {
          maxDepth = std::max(maxDepth, depthMap[operand]);
        }
      }

      // Specific handling based on operation type
      if (auto mulOp = dyn_cast<openfhe::MulOp>(operation)) {
        // Multiplication increases depth by 1
        depthMap[mulOp.getResult()] = maxDepth + 1;
      } else if (auto mulPlainOp = dyn_cast<openfhe::MulPlainOp>(operation)) {
        // Multiplication by plaintext also increases depth by 1
        depthMap[mulPlainOp.getResult()] = maxDepth + 1;
      } else {
        // All other operations maintain depth
        for (Value result : operation->getResults()) {
          depthMap[result] = maxDepth;
        }
      }
    });

    // Second pass: Find the minimum depth at which each rotation key is used
    op->walk([&](openfhe::RotOp rotOp) {
      // Get the input depth
      Value input = rotOp.getCiphertext();
      unsigned inputDepth = depthMap.count(input) > 0 ? depthMap[input] : 0;

      // Get the key
      Value key = rotOp.getEvalKey();

      // Find the deserialize op that produced this key
      if (auto definingOp = key.getDefiningOp()) {
        if (auto deserOp = dyn_cast<openfhe::DeserializeKeyOp>(definingOp)) {
          // Update the minimum depth for this deserialize op
          auto it = deserializeDepthMap.find(deserOp);
          if (it == deserializeDepthMap.end() || inputDepth < it->second) {
            deserializeDepthMap[deserOp] = inputDepth;
          }

          // Extract the rotation index
          int64_t index = -1;
          if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
            index = indexAttr.getInt();
          } else if (auto keyType =
                         dyn_cast<openfhe::EvalKeyType>(key.getType())) {
            if (auto indexAttr = keyType.getRotationIndex()) {
              index = indexAttr.getInt();
            }
          }

          // Record all depths at which this index is used
          if (index != -1) {
            indexToDepths[index].insert(inputDepth);
          }
        }
      }
    });

    // Third pass: Add depth attributes to deserialize operations
    op->walk([&](openfhe::DeserializeKeyOp deserOp) {
      // Get the index
      int64_t index = -1;
      if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
        index = indexAttr.getInt();
      } else if (auto keyType = dyn_cast<openfhe::EvalKeyType>(
                     deserOp.getEvalKey().getType())) {
        if (auto indexAttr = keyType.getRotationIndex()) {
          index = indexAttr.getInt();
        }
      }

      if (index != -1) {
        // Set the depth attribute
        unsigned depth = 0;
        auto it = deserializeDepthMap.find(deserOp);
        if (it != deserializeDepthMap.end()) {
          depth = it->second;
        }

        // Create attributes using IntegerAttr directly
        deserOp->setAttr("key_depth",
                         IntegerAttr::get(IntegerType::get(ctx, 64), depth));

        llvm::errs() << "Setting depth " << depth << " for key " << index
                     << "\n";
      }
    });

    // Collect information for GenRotKeyDepth operations
    std::map<unsigned, std::set<int64_t>> depthToIndices;
    for (auto &[index, depths] : indexToDepths) {
      for (unsigned depth : depths) {
        depthToIndices[depth].insert(index);
      }
    }

    // Make sure all indices are included at depth 0
    for (auto &[index, _] : indexToDepths) {
      depthToIndices[0].insert(index);
    }

    // Find the first GenRotKeyOp to get its location and context for new ops
    bool foundGenRotKeyOp = false;
    openfhe::GenRotKeyOp templateOp;

    // Find the first GenRotKeyOp in the module
    op->walk([&](openfhe::GenRotKeyOp genRotOp) {
      if (!foundGenRotKeyOp) {
        foundGenRotKeyOp = true;
        templateOp = genRotOp;
      }
    });

    // If found, add GenRotKeyDepth operations for each depth
    if (foundGenRotKeyOp) {
      // Find all GenRotKeyOp operations to replace
      SmallVector<openfhe::GenRotKeyOp, 4> genRotOpsToReplace;
      op->walk([&](openfhe::GenRotKeyOp genRotOp) {
        genRotOpsToReplace.push_back(genRotOp);
      });

      if (!genRotOpsToReplace.empty()) {
        // Use the first GenRotKeyOp as the insertion point
        auto firstGenRotOp = genRotOpsToReplace[0];
        OpBuilder builder(firstGenRotOp);

        // For each depth, create a GenRotKeyDepth operation
        for (auto &[depth, indicesSet] : depthToIndices) {
          // Convert set to vector and sort
          SmallVector<int64_t, 8> indices(indicesSet.begin(), indicesSet.end());
          std::sort(indices.begin(), indices.end());

          // Create integer attributes for indices
          SmallVector<Attribute, 8> indexAttrs;
          for (int64_t idx : indices) {
            indexAttrs.push_back(builder.getI64IntegerAttr(idx));
          }

          // Create array attribute
          auto indicesAttr = builder.getArrayAttr(indexAttrs);

          // Create the GenRotKeyDepth operation
          builder.create<openfhe::GenRotKeyDepthOp>(
              templateOp.getLoc(), templateOp.getCryptoContext(),
              templateOp.getPrivateKey(), indicesAttr,
              builder.getI64IntegerAttr(depth));

          llvm::errs() << "Created GenRotKeyDepth operation with "
                       << indices.size() << " indices at depth " << depth
                       << "\n";
        }

        // Remove the original GenRotKeyOp operations
        for (auto genRotOp : genRotOpsToReplace) {
          genRotOp->erase();
        }
      }
    }

    // Print summary of key depths
    llvm::errs() << "\n===== Key Depth Analysis =====\n";

    // Print each rotation index and the depths it's used at
    std::vector<int64_t> allIndices;
    for (auto &[index, depths] : indexToDepths) {
      allIndices.push_back(index);
    }

    std::sort(allIndices.begin(), allIndices.end());

    for (int64_t index : allIndices) {
      const auto &depths = indexToDepths[index];

      llvm::errs() << "Key " << index << " used at depths: [";
      bool first = true;
      for (unsigned depth : depths) {
        if (!first) llvm::errs() << ", ";
        llvm::errs() << depth;
        first = false;
      }
      llvm::errs() << "]\n";
    }

    llvm::errs() << "=================================\n\n";
  }
};

}  // namespace

std::unique_ptr<Pass> createKeyCompressionPass() {
  return std::make_unique<KeyCompression>();
}

}  // namespace heir
}  // namespace mlir
