#include "lib/Transforms/FHEFunctionOutlining/FHEFunctionOutlining.h"

#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "llvm/include/llvm/ADT/DenseMap.h"
#include "llvm/include/llvm/ADT/SmallVector.h"
#include "llvm/include/llvm/Support/Debug.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/IRMapping.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/Support/LLVM.h"

#define DEBUG_TYPE "fhe-function-outlining"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_FHEFUNCTIONOUTLINING
#include "lib/Transforms/FHEFunctionOutlining/FHEFunctionOutlining.h.inc"

// Simple node for DAG analysis
struct DAGNode {
  Operation *op;
  std::vector<DAGNode *> predecessors;
  std::vector<DAGNode *> successors;
  int index;

  DAGNode(Operation *operation, int idx) : op(operation), index(idx) {}
};

// Helper to check if operation should be included in analysis
static bool isRelevantOperation(Operation *op) {
  // Skip function calls and return ops, but keep constants for now
  if (isa<func::CallOp>(op) || isa<func::ReturnOp>(op)) {
    return false;
  }

  // Include CKKS operations and other computational ops
  if (isa<ckks::LinearTransformOp>(op) || isa<ckks::MulOp>(op) ||
      isa<ckks::AddOp>(op) || isa<ckks::SubOp>(op) || isa<ckks::RotateOp>(op) ||
      isa<ckks::MulPlainOp>(op) || isa<ckks::AddPlainOp>(op) ||
      isa<ckks::BootstrapOp>(op) || isa<ckks::ChebyshevOp>(op)) {
    return true;
  }

  // Include LWE operations
  if (op->getDialect()->getNamespace() == "lwe") {
    return true;
  }

  // Include constants - we'll filter out unused ones later
  if (isa<arith::ConstantOp>(op)) {
    return true;
  }

  return false;
}

// Helper to get the operation weight (complexity) for balancing
static int getOperationWeight(Operation *op) {
  // For linear transform, weight by diagonal_count since it expands to many ops
  if (auto linTransOp = dyn_cast<ckks::LinearTransformOp>(op)) {
    int32_t diagonalCount = linTransOp.getDiagonalCount();

    // More accurate estimation based on actual lowering:
    // Each diagonal creates: 1 rotation + 1 mul_plain + tree reduction
    // Tree reduction adds log2(diagonal_count) operations
    // But empirically, we're seeing even more expansion than this

    // Use a more aggressive multiplier for very large diagonal counts
    int baseWeight = diagonalCount * 3;  // rotation + mul + accumulation
    int treeReduction =
        diagonalCount > 1 ? llvm::Log2_32(diagonalCount) * 2 : 0;

    // Add extra weight for very large diagonal counts (they seem to expand
    // more)
    int extraWeight = 0;
    if (diagonalCount > 1000) {
      extraWeight =
          diagonalCount / 10;  // Additional 10% for very large transforms
    } else if (diagonalCount > 500) {
      extraWeight = diagonalCount / 20;  // Additional 5% for large transforms
    }

    int totalWeight = baseWeight + treeReduction + extraWeight;

    LLVM_DEBUG(llvm::dbgs()
               << "LinearTransform weight calculation: diagonal_count="
               << diagonalCount << " -> weight=" << totalWeight << "\n");

    return totalWeight;
  }

  // All other relevant operations count as 1
  if (isRelevantOperation(op)) {
    return 1;
  }

  return 0;
}

// Simple DAG analyzer for finding split points
class SimpleDAGAnalyzer {
 private:
  std::vector<std::unique_ptr<DAGNode>> nodes;
  llvm::DenseMap<Operation *, DAGNode *> opToNode;
  std::vector<int>
      cumulativeWeights;  // Track cumulative weights for balanced splitting

 public:
  void analyze(func::FuncOp funcOp) {
    // First pass: Create nodes for all relevant operations
    int index = 0;
    int cumulativeWeight = 0;
    funcOp.walk([&](Operation *op) {
      if (isRelevantOperation(op)) {
        auto node = std::make_unique<DAGNode>(op, index++);
        opToNode[op] = node.get();
        nodes.push_back(std::move(node));

        // Track cumulative weights for balanced splitting
        int opWeight = getOperationWeight(op);
        cumulativeWeight += opWeight;
        cumulativeWeights.push_back(cumulativeWeight);

        // Debug output for linear transforms to understand the distribution
        if (auto linTransOp = dyn_cast<ckks::LinearTransformOp>(op)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Operation " << (index - 1) << ": LinearTransform"
                     << " diagonal_count=" << linTransOp.getDiagonalCount()
                     << " weight=" << opWeight << "\n");
        }
      }
    });

    // Build DAG edges
    for (auto &node : nodes) {
      for (Value operand : node->op->getOperands()) {
        if (auto defOp = operand.getDefiningOp()) {
          if (auto predNode = opToNode.lookup(defOp)) {
            node->predecessors.push_back(predNode);
            predNode->successors.push_back(node.get());
          }
        }
      }
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Built DAG with " << nodes.size() << " nodes, total weight: "
               << (cumulativeWeights.empty() ? 0 : cumulativeWeights.back())
               << "\n");
  }

  // First find all valid split points, then select balanced ones
  std::vector<int> findBalancedSplitPoints(int numSegments) {
    if (nodes.empty() || numSegments <= 1) {
      return {};
    }

    // Step 1: Find all valid split points (clean cuts)
    std::vector<std::pair<int, int>> validSplits;  // (index, cross_deps)

    LLVM_DEBUG(llvm::dbgs() << "Step 1: Finding all valid split points...\n");

    for (int i = 1; i < static_cast<int>(nodes.size()) - 1; i++) {
      int crossDeps = countCrossDependencies(i);

      // Count how many of the cross-dependencies are ciphertext vs plaintext
      int ciphertextDeps = 0;
      int plaintextDeps = 0;

      for (int j = i; j < static_cast<int>(nodes.size()); j++) {
        if (auto node = getNode(j)) {
          for (auto pred : node->predecessors) {
            if (pred->index < i) {
              // This is a cross-dependency
              if (isa<arith::ConstantOp>(pred->op)) {
                plaintextDeps++;  // Constants are easy to copy
              } else {
                // Check if this produces a ciphertext or plaintext
                if (!pred->op->getResults().empty()) {
                  Type resultType = pred->op->getResult(0).getType();
                  if (isa<lwe::NewLWECiphertextType>(resultType)) {
                    ciphertextDeps++;
                  } else {
                    plaintextDeps++;  // Plaintext operations can be copied
                  }
                }
              }
            }
          }
        }
      }

      // A split is valid if it has at most 1 ciphertext dependency (the main
      // flow)
      if (ciphertextDeps <= 1) {
        validSplits.push_back({i, crossDeps});
        LLVM_DEBUG(llvm::dbgs() << "  Valid split at index " << i
                                << " (ct_deps: " << ciphertextDeps
                                << ", pt_deps: " << plaintextDeps << ")\n");
      }
    }

    if (validSplits.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No valid split points found!\n");
      return {};
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Found " << validSplits.size() << " valid split points\n");

    // Step 2: Select balanced split points from valid candidates
    // Ensure each segment gets at least one linear transform
    std::vector<int> splitPoints;
    int totalWeight = cumulativeWeights.empty() ? 0 : cumulativeWeights.back();
    int targetWeightPerSegment = totalWeight / numSegments;

    LLVM_DEBUG(
        llvm::dbgs()
        << "Step 2: Selecting balanced splits (target weight per segment: "
        << targetWeightPerSegment << ")\n");

    // Helper function to check if a segment contains at least one linear
    // transform
    auto segmentHasLinearTransform = [&](int startIdx, int endIdx) -> bool {
      for (int i = startIdx; i < endIdx; i++) {
        if (auto node = getNode(i)) {
          if (isa<ckks::LinearTransformOp>(node->op)) {
            return true;
          }
        }
      }
      return false;
    };

    for (int segment = 1; segment < numSegments; segment++) {
      int targetWeight = segment * targetWeightPerSegment;

      // Find the valid split point closest to our target weight that ensures
      // both the current segment and the next segment have linear transforms
      int bestSplit = -1;
      int minWeightDiff = INT_MAX;

      for (auto [splitIndex, crossDeps] : validSplits) {
        // Skip if this would create duplicate or out-of-order splits
        if (!splitPoints.empty() && splitIndex <= splitPoints.back()) continue;
        if (splitIndex >= static_cast<int>(nodes.size()) - 1) continue;

        // Check if the current segment (up to this split) has a linear
        // transform
        int segmentStart = splitPoints.empty() ? 0 : splitPoints.back();
        bool currentSegmentHasLinTrans =
            segmentHasLinearTransform(segmentStart, splitIndex);

        // Check if the remaining operations (after this split) have at least
        // one linear transform BUT only if this isn't the last segment we're
        // creating
        bool remainingHasLinTrans = true;  // Default to true
        int remainingSegments = numSegments - segment;
        if (remainingSegments > 1) {
          // Only check remaining segments if we're creating more than 1
          // additional segment
          remainingHasLinTrans = segmentHasLinearTransform(
              splitIndex, static_cast<int>(nodes.size()));
        }

        if (!currentSegmentHasLinTrans || !remainingHasLinTrans) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  Skipping split at " << splitIndex
                     << " (current_seg_has_lt: " << currentSegmentHasLinTrans
                     << ", remaining_has_lt: " << remainingHasLinTrans
                     << ", remaining_segments: " << remainingSegments << ")\n");
          continue;
        }

        int splitWeight = splitIndex < cumulativeWeights.size()
                              ? cumulativeWeights[splitIndex]
                              : 0;
        int weightDiff = std::abs(splitWeight - targetWeight);

        if (weightDiff < minWeightDiff) {
          minWeightDiff = weightDiff;
          bestSplit = splitIndex;
        }
      }

      if (bestSplit != -1) {
        splitPoints.push_back(bestSplit);
        int splitWeight = bestSplit < cumulativeWeights.size()
                              ? cumulativeWeights[bestSplit]
                              : 0;

        // Verify this segment has a linear transform
        int segmentStart =
            splitPoints.size() > 1 ? splitPoints[splitPoints.size() - 2] : 0;
        bool hasLinTrans = segmentHasLinearTransform(segmentStart, bestSplit);

        LLVM_DEBUG(llvm::dbgs()
                   << "Selected split point " << segment << ": index "
                   << bestSplit << " (weight: " << splitWeight << ", target: "
                   << targetWeight << ", diff: " << minWeightDiff
                   << ", has_linear_transform: " << hasLinTrans << ")\n");
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "Could not find valid split point for segment " << segment
                   << " that ensures linear transform in each segment\n");
        break;
      }
    }

    return splitPoints;
  }

  // Use the balanced approach as the main split method
  std::vector<int> findSplitPoints(int splitCount) {
    return findBalancedSplitPoints(splitCount);
  }

  int countCrossDependencies(int splitPoint) {
    int count = 0;
    for (int i = splitPoint; i < static_cast<int>(nodes.size()); i++) {
      if (auto node = getNode(i)) {
        for (auto pred : node->predecessors) {
          if (pred->index < splitPoint) {
            count++;
          }
        }
      }
    }
    return count;
  }

  int size() const { return nodes.size(); }

  DAGNode *getNode(int index) const {
    return index >= 0 && index < static_cast<int>(nodes.size())
               ? nodes[index].get()
               : nullptr;
  }

  // Get constants that are directly used by operations in a segment
  llvm::DenseSet<Operation *> getSegmentConstants(int startIdx, int endIdx) {
    llvm::DenseSet<Operation *> segmentConstants;

    for (int i = startIdx; i < endIdx; i++) {
      if (auto node = getNode(i)) {
        Operation *op = node->op;
        if (!isa<arith::ConstantOp>(op)) {
          for (Value operand : op->getOperands()) {
            if (auto defOp = operand.getDefiningOp()) {
              if (isa<arith::ConstantOp>(defOp)) {
                segmentConstants.insert(defOp);
              }
            }
          }
        }
      }
    }

    return segmentConstants;
  }
};

struct FHEFunctionOutlining
    : impl::FHEFunctionOutliningBase<FHEFunctionOutlining> {
  using impl::FHEFunctionOutliningBase<
      FHEFunctionOutlining>::FHEFunctionOutliningBase;

  void runOnOperation() override {
    auto funcOp = getOperation();

    LLVM_DEBUG(llvm::dbgs()
               << "=== FHE Function Outlining processing function: "
               << funcOp.getName() << " ===\n");

    // First, verify the function is valid
    if (funcOp.getBody().empty()) {
      LLVM_DEBUG(llvm::dbgs() << "Function has empty body, skipping\n");
      return;
    }

    // Skip small functions or helper functions
    std::string funcName = funcOp.getName().str();
    if (funcName.find("__generate_crypto_context") != std::string::npos ||
        funcName.find("__configure_crypto_context") != std::string::npos ||
        funcName.find("_segment_") != std::string::npos ||
        (funcName.length() >= 5 &&
         funcName.substr(funcName.length() - 5) == "_main")) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping helper/outlined function\n");
      return;
    }

    // Check if function is already outlined
    bool alreadyOutlined = false;
    funcOp.walk([&](func::CallOp callOp) {
      if (callOp.getCallee().str().find("_segment_") != std::string::npos) {
        alreadyOutlined = true;
        LLVM_DEBUG(llvm::dbgs() << "Function appears to be already outlined\n");
      }
    });

    if (alreadyOutlined) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping already outlined function\n");
      funcOp.emitWarning()
          << "Function " << funcOp.getName()
          << " appears to be already outlined. Please use a fresh input file.";
      return;
    }

    // Analyze the function to build DAG
    SimpleDAGAnalyzer analyzer;
    analyzer.analyze(funcOp);

    if (analyzer.size() < splitCount) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Function has too few operations (" << analyzer.size()
                 << ") for splitting into " << splitCount << " parts\n");
      return;
    }

    // Find split points
    auto splitPoints = analyzer.findSplitPoints(splitCount);
    if (splitPoints.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No valid split points found\n");
      return;
    }

    // Create segments
    std::vector<std::pair<int, int>> segments;
    int start = 0;
    for (int splitPoint : splitPoints) {
      segments.emplace_back(start, splitPoint);
      start = splitPoint;
    }
    segments.emplace_back(start, analyzer.size());

    LLVM_DEBUG(llvm::dbgs() << "Created " << segments.size() << " segments\n");

    // Get function types for segments
    auto [segmentInputTypes, segmentOutputTypes] =
        determineSegmentTypes(funcOp, analyzer, segments);

    // Create outlined functions - with better error handling
    OpBuilder builder(&getContext());
    std::string baseName = funcOp.getName().str();
    std::vector<func::FuncOp> segmentFunctions;

    LLVM_DEBUG(llvm::dbgs() << "Attempting to create " << segments.size()
                            << " segment functions\n");

    for (size_t i = 0; i < segments.size(); i++) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Creating segment " << i << " from indices "
                 << segments[i].first << " to " << segments[i].second << "\n");

      auto segmentFunc = createSegmentFunctionWithTypes(
          builder, funcOp, analyzer, segments[i], static_cast<int>(i), baseName,
          segmentInputTypes[i], segmentOutputTypes[i]);

      if (segmentFunc) {
        segmentFunctions.push_back(segmentFunc);
        LLVM_DEBUG(llvm::dbgs() << "✅ Successfully created segment function: "
                                << segmentFunc.getName() << "\n");
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "❌ Failed to create segment function " << i << "\n");
        // If any segment fails, abort the entire outlining to avoid broken MLIR
        funcOp.emitError() << "Failed to create segment " << i
                           << " during outlining";

        // Clean up any partial segment functions that were created
        for (auto partialFunc : segmentFunctions) {
          partialFunc.erase();
        }
        return;
      }
    }

    // Verify we created the expected number of segments
    if (segmentFunctions.size() != segments.size()) {
      LLVM_DEBUG(llvm::dbgs() << "❌ Expected " << segments.size()
                              << " segments but only created "
                              << segmentFunctions.size() << "\n");
      funcOp.emitError() << "Outlining failed: expected " << segments.size()
                         << " segments but only created "
                         << segmentFunctions.size();
      return;
    }

    // Only create main dispatcher if we successfully created ALL segment
    // functions
    if (!segmentFunctions.empty() &&
        segmentFunctions.size() == segments.size()) {
      createMainFunction(builder, funcOp, segmentFunctions, baseName);
      funcOp.setPrivate();
      funcOp.setName(baseName + "_original");
      LLVM_DEBUG(llvm::dbgs() << "✅ Successfully completed outlining\n");
    }
  }

  std::pair<std::vector<Type>, std::vector<Type>> determineSegmentTypes(
      func::FuncOp funcOp, SimpleDAGAnalyzer &analyzer,
      const std::vector<std::pair<int, int>> &segments) {
    std::vector<Type> inputTypes;
    std::vector<Type> outputTypes;

    // First pass: determine all output types
    for (size_t i = 0; i < segments.size(); i++) {
      auto segment = segments[i];
      Type outputType;

      // Find the last computational operation in this segment
      for (int j = segment.second - 1; j >= segment.first; j--) {
        if (auto node = analyzer.getNode(j)) {
          if (!isa<arith::ConstantOp>(node->op) &&
              !node->op->getResults().empty()) {
            outputType = node->op->getResult(0).getType();
            break;
          }
        }
      }

      // Fallback: use the original function's return type
      if (!outputType) {
        outputType = funcOp.getResultTypes()[0];
      }

      outputTypes.push_back(outputType);
    }

    // Second pass: determine input types based on output types
    for (size_t i = 0; i < segments.size(); i++) {
      Type inputType;

      if (i == 0) {
        // First segment: input type is the original function's first argument
        inputType = funcOp.getArgumentTypes()[0];
      } else {
        // Later segments: input type is the previous segment's output type
        inputType = outputTypes[i - 1];
      }

      inputTypes.push_back(inputType);
    }

    return std::make_pair(inputTypes, outputTypes);
  }

  func::FuncOp createSegmentFunctionWithTypes(OpBuilder &builder,
                                              func::FuncOp originalFunc,
                                              SimpleDAGAnalyzer &analyzer,
                                              std::pair<int, int> segment,
                                              int segmentIndex,
                                              const std::string &baseName,
                                              Type inputType, Type outputType) {
    int startIdx = segment.first;
    int endIdx = segment.second;

    LLVM_DEBUG(llvm::dbgs()
               << "Creating segment " << segmentIndex << " with operations "
               << startIdx << " to " << endIdx
               << " (range size: " << (endIdx - startIdx) << ")\n");

    if (startIdx >= endIdx) {
      LLVM_DEBUG(llvm::dbgs()
                 << "❌ Invalid segment range: startIdx (" << startIdx
                 << ") >= endIdx (" << endIdx << ")\n");
      return nullptr;
    }

    // Special handling for single-operation segments
    if (endIdx - startIdx == 1) {
      LLVM_DEBUG(llvm::dbgs() << "Single-operation segment detected\n");
    }

    // Get all operations for this segment in their original order
    std::vector<Operation *> allOpsToClone;
    auto segmentConstants = analyzer.getSegmentConstants(startIdx, endIdx);

    LLVM_DEBUG(llvm::dbgs() << "Segment " << segmentIndex << " needs "
                            << segmentConstants.size() << " constants\n");

    originalFunc.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op == originalFunc.getOperation()) {
        return WalkResult::advance();
      }

      // Check if this is a constant needed by this segment
      if (isa<arith::ConstantOp>(op) && segmentConstants.contains(op)) {
        allOpsToClone.push_back(op);
        LLVM_DEBUG(llvm::dbgs()
                   << "  Adding constant to segment " << segmentIndex << "\n");
        return WalkResult::advance();
      }

      // Check if this is a computational operation in this segment
      bool foundInSegment = false;
      for (int i = startIdx; i < endIdx; i++) {
        if (auto node = analyzer.getNode(i)) {
          if (node->op == op && !isa<arith::ConstantOp>(op)) {
            allOpsToClone.push_back(op);
            foundInSegment = true;
            LLVM_DEBUG(llvm::dbgs()
                       << "  Adding operation " << i << " to segment "
                       << segmentIndex << ": " << op->getName() << "\n");
            break;
          }
        }
      }

      // Debug: If this is a single-operation segment and we haven't found
      // anything, investigate
      if (!foundInSegment && endIdx - startIdx == 1 &&
          startIdx < analyzer.size()) {
        if (auto node = analyzer.getNode(startIdx)) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  DEBUG: Single-op segment " << segmentIndex
                     << " should contain operation " << startIdx
                     << " which is: " << node->op->getName() << "\n");
        }
      }

      return WalkResult::advance();
    });

    LLVM_DEBUG(llvm::dbgs()
               << "Segment " << segmentIndex << " collected "
               << allOpsToClone.size() << " operations to clone\n");

    if (allOpsToClone.empty()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "❌ No operations to clone in segment " << segmentIndex
                 << " - this suggests a problem with the split points\n");
      return nullptr;
    }

    // Create function
    auto savedInsertionPoint = builder.saveInsertionPoint();
    builder.setInsertionPointAfter(originalFunc);

    auto segmentFuncType = builder.getFunctionType({inputType}, {outputType});
    std::string segmentName =
        baseName + "_segment_" + std::to_string(segmentIndex);
    auto segmentFunc = builder.create<func::FuncOp>(
        originalFunc.getLoc(), segmentName, segmentFuncType);

    Block *segmentBody = segmentFunc.addEntryBlock();
    builder.setInsertionPointToStart(segmentBody);

    // Value mapping
    IRMapping valueMap;
    Value inputValue = segmentFunc.getArgument(0);

    // For the first segment, map original function arguments
    if (segmentIndex == 0) {
      for (auto [origArg, newArg] :
           llvm::zip(originalFunc.getArguments(), segmentFunc.getArguments())) {
        valueMap.map(origArg, newArg);
      }
    } else {
      // For subsequent segments with clean cuts, there should be exactly one
      // ciphertext flow and possibly some constants that need to be copied
      bool foundCiphertextMapping = false;

      for (Operation *op : allOpsToClone) {
        if (isa<arith::ConstantOp>(op))
          continue;  // Handle constants separately

        for (Value operand : op->getOperands()) {
          if (auto defOp = operand.getDefiningOp()) {
            // Check if this operand is defined before our segment starts
            bool isFromPreviousSegment = true;
            for (Operation *segmentOp : allOpsToClone) {
              if (segmentOp == defOp) {
                isFromPreviousSegment = false;
                break;
              }
            }

            if (isFromPreviousSegment && !isa<arith::ConstantOp>(defOp)) {
              // This should be the main ciphertext flow - map to function
              // argument
              Type operandType = operand.getType();
              if (isa<lwe::NewLWECiphertextType>(operandType) &&
                  !foundCiphertextMapping) {
                valueMap.map(operand, inputValue);
                foundCiphertextMapping = true;
                LLVM_DEBUG(llvm::dbgs()
                           << "  Mapped ciphertext flow to input for segment "
                           << segmentIndex << "\n");
                break;
              }
            }
          }
        }
        if (foundCiphertextMapping) break;
      }
    }

    // Clone operations in original order
    for (Operation *op : allOpsToClone) {
      // Handle any remaining unmapped operands (should mostly be constants)
      for (auto operand : op->getOperands()) {
        if (auto defOp = operand.getDefiningOp()) {
          if (valueMap.lookupOrNull(operand) == nullptr) {
            // Check if this operand is from outside this segment
            bool isFromPreviousSegment = true;
            for (Operation *segmentOp : allOpsToClone) {
              if (segmentOp == defOp) {
                isFromPreviousSegment = false;
                break;
              }
            }

            if (isFromPreviousSegment) {
              // With clean cuts, external operands should be constants or
              // plaintext Clone them into this segment
              auto clonedOp = builder.clone(*defOp);
              valueMap.map(operand, clonedOp->getResult(0));
              // Also map the original operation's results
              for (auto [origResult, newResult] :
                   llvm::zip(defOp->getResults(), clonedOp->getResults())) {
                valueMap.map(origResult, newResult);
              }
              LLVM_DEBUG(llvm::dbgs()
                         << "  Cloned external operation in segment "
                         << segmentIndex << "\n");
            }
          }
        }
      }

      auto clonedOp = builder.clone(*op, valueMap);

      // Update value mapping with results
      for (auto [origResult, newResult] :
           llvm::zip(op->getResults(), clonedOp->getResults())) {
        valueMap.map(origResult, newResult);
      }
    }

    // Find the actual output value
    Value returnValue = inputValue;  // Default fallback

    for (auto it = allOpsToClone.rbegin(); it != allOpsToClone.rend(); ++it) {
      Operation *op = *it;
      if (!isa<arith::ConstantOp>(op) && !op->getResults().empty()) {
        Value lastResult = op->getResult(0);
        Value mappedResult = valueMap.lookupOrNull(lastResult);
        if (mappedResult && mappedResult.getType() == outputType) {
          returnValue = mappedResult;
          break;
        }
      }
    }

    builder.create<func::ReturnOp>(originalFunc.getLoc(), returnValue);
    builder.restoreInsertionPoint(savedInsertionPoint);

    LLVM_DEBUG(llvm::dbgs() << "✅ Successfully created segment function: "
                            << segmentName << "\n");
    return segmentFunc;
  }

  void createMainFunction(OpBuilder &builder, func::FuncOp originalFunc,
                          const std::vector<func::FuncOp> &segmentFunctions,
                          const std::string &baseName) {
    auto savedInsertionPoint = builder.saveInsertionPoint();
    builder.setInsertionPointAfter(originalFunc);

    std::string mainFuncName = baseName + "_main";
    auto mainFuncType = originalFunc.getFunctionType();
    auto mainFunc = builder.create<func::FuncOp>(originalFunc.getLoc(),
                                                 mainFuncName, mainFuncType);

    Block *mainBody = mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(mainBody);

    // Chain calls to segment functions
    Value currentValue = mainFunc.getArgument(0);

    for (auto segmentFunc : segmentFunctions) {
      auto callOp = builder.create<func::CallOp>(
          originalFunc.getLoc(), segmentFunc.getName(),
          segmentFunc.getResultTypes(), ValueRange{currentValue});
      currentValue = callOp.getResult(0);
    }

    builder.create<func::ReturnOp>(originalFunc.getLoc(), currentValue);
    builder.restoreInsertionPoint(savedInsertionPoint);

    LLVM_DEBUG(llvm::dbgs()
               << "Created main dispatcher function: " << mainFuncName << "\n");
  }
};

std::unique_ptr<Pass> createFHEFunctionOutliningPass() {
  return std::make_unique<FHEFunctionOutlining>();
}

}  // namespace heir
}  // namespace mlir
