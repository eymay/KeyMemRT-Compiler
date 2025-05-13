#include "RotationDepthAnalysis.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Support/LLVM.h"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ROTATIONDEPTHANALYSIS
#include "lib/Transforms/RotationDepthAnalysis/RotationDepthAnalysis.h.inc"

namespace {

// Helper function to convert a set of indices to a range string representation
std::string indicesToRangeString(const std::set<int64_t> &indices) {
  if (indices.empty()) {
    return "[]";
  }

  std::vector<std::pair<int64_t, int64_t>> ranges;
  auto it = indices.begin();
  int64_t start = *it;
  int64_t end = start;

  ++it;
  while (it != indices.end()) {
    if (*it == end + 1) {
      // Extend current range
      end = *it;
    } else {
      // Finish current range and start a new one
      ranges.push_back({start, end});
      start = *it;
      end = start;
    }
    ++it;
  }

  // Add final range
  ranges.push_back({start, end});

  // Convert ranges to string
  std::stringstream ss;
  for (size_t i = 0; i < ranges.size(); ++i) {
    auto [s, e] = ranges[i];
    if (s == e) {
      ss << s;
    } else {
      ss << s << "-" << e;
    }

    if (i < ranges.size() - 1) {
      ss << ", ";
    }
  }

  return ss.str();
}

/// This pass analyzes the multiplicative depth of rotation operations in
/// OpenFHE programs. It tracks operations through the program, calculating the
/// multiplicative depth at each point, and counts the number of rotations that
/// occur at each depth while also tracking rotation indices.
struct RotationDepthAnalysis
    : impl::RotationDepthAnalysisBase<RotationDepthAnalysis> {
  void runOnOperation() override {
    Operation *op = getOperation();

    // Maps each Value to its current multiplicative depth
    llvm::DenseMap<Value, unsigned> depthMap;

    // Maps multiplicative depth to count of rotation operations at that depth
    std::map<unsigned, unsigned> rotationsAtDepth;

    // Maps multiplicative depth to the set of rotation indices used at that
    // depth
    std::map<unsigned, std::set<int64_t>> indicesAtDepth;

    // Initialize depth for block arguments
    op->walk([&](Block *block) {
      for (BlockArgument arg : block->getArguments()) {
        depthMap[arg] = 0;
      }
    });

    // Walk through all operations in a topological order
    op->walk([&](Operation *op) {
      // Calculate max depth of operands
      unsigned maxDepth = 0;
      for (Value operand : op->getOperands()) {
        if (depthMap.count(operand) > 0) {
          maxDepth = std::max(maxDepth, depthMap[operand]);
        }
      }

      // Specific handling based on operation type
      if (auto mulOp = dyn_cast<openfhe::MulOp>(op)) {
        // Multiplication increases depth by 1
        depthMap[mulOp.getResult()] = maxDepth + 1;
      } else if (auto mulPlainOp = dyn_cast<openfhe::MulPlainOp>(op)) {
        // Multiplication by plaintext also increases depth by 1
        depthMap[mulPlainOp.getResult()] = maxDepth + 1;
      } else if (auto rotOp = dyn_cast<openfhe::RotOp>(op)) {
        // Rotations maintain the depth but we count them
        depthMap[rotOp.getResult()] = maxDepth;

        // Count this rotation at its depth
        rotationsAtDepth[maxDepth]++;

        // Try to extract the rotation index from the operation
        // First, check if the evalKey operand is a direct result of a
        // DeserializeKeyOp
        Value keyOperand = rotOp.getEvalKey();
        for (auto user : keyOperand.getUsers()) {
          if (auto deserOp = dyn_cast<openfhe::DeserializeKeyOp>(user)) {
            // Look for the index attribute from the deserialize operation
            if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
              int64_t index = indexAttr.getInt();
              indicesAtDepth[maxDepth].insert(index);
              break;
            }
          }
        }

        // If not found from users, check if the evalKey originated from a
        // deserialize op
        if (auto definingOp = keyOperand.getDefiningOp()) {
          if (auto deserOp = dyn_cast<openfhe::DeserializeKeyOp>(definingOp)) {
            // Look for the index attribute from the deserialize operation
            if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
              int64_t index = indexAttr.getInt();
              indicesAtDepth[maxDepth].insert(index);
            }
          }
        }

        // If the index comes from the type, try to extract it
        if (auto evalKeyType =
                dyn_cast<openfhe::EvalKeyType>(rotOp.getEvalKey().getType())) {
          if (auto indexAttr = evalKeyType.getRotationIndex()) {
            int64_t index = indexAttr.getInt();
            indicesAtDepth[maxDepth].insert(index);
          }
        }
      } else {
        // All other operations maintain depth
        for (Value result : op->getResults()) {
          depthMap[result] = maxDepth;
        }
      }
    });

    // Print the analysis results
    llvm::outs() << "\n===== Rotation Depth Analysis =====\n";

    if (rotationsAtDepth.empty()) {
      llvm::outs() << "No rotation operations found.\n";
    } else {
      llvm::outs() << "Rotation operations by multiplicative depth:\n";

      unsigned totalRotations = 0;
      for (const auto &[depth, count] : rotationsAtDepth) {
        llvm::outs() << "  Depth " << depth << ": " << count << " rotations";

        // Print indices if available
        auto it = indicesAtDepth.find(depth);
        if (it != indicesAtDepth.end() && !it->second.empty()) {
          llvm::outs() << " (indices: " << indicesToRangeString(it->second)
                       << ")";
        }

        llvm::outs() << "\n";
        totalRotations += count;
      }

      llvm::outs() << "Total rotations: " << totalRotations << "\n";

      // Print a summary of all distinct rotation indices used
      std::set<int64_t> allIndices;
      for (const auto &[depth, indices] : indicesAtDepth) {
        allIndices.insert(indices.begin(), indices.end());
      }

      if (!allIndices.empty()) {
        llvm::outs() << "All rotation indices used: "
                     << indicesToRangeString(allIndices) << "\n";
      }
    }

    llvm::outs() << "===================================\n\n";
  }
};

}  // namespace

}  // namespace heir
}  // namespace mlir
