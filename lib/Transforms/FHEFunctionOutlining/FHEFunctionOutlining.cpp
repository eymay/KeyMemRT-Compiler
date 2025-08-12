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
  // Skip constants, function calls, and other non-computational ops
  if (isa<arith::ConstantOp>(op) || isa<func::CallOp>(op) ||
      isa<func::ReturnOp>(op)) {
    return false;
  }

  // Include CKKS operations and other computational ops
  return isa<ckks::LinearTransformOp>(op) || isa<ckks::MulOp>(op) ||
         isa<ckks::AddOp>(op) || isa<ckks::SubOp>(op) ||
         isa<ckks::RotateOp>(op) || isa<ckks::MulPlainOp>(op) ||
         isa<ckks::AddPlainOp>(op);
}

// Simple DAG analyzer for finding split points
class SimpleDAGAnalyzer {
 private:
  std::vector<std::unique_ptr<DAGNode>> nodes;
  llvm::DenseMap<Operation *, DAGNode *> opToNode;

 public:
  void analyze(func::FuncOp funcOp) {
    // Create nodes for relevant operations
    int index = 0;
    funcOp.walk([&](Operation *op) {
      if (isRelevantOperation(op)) {
        auto node = std::make_unique<DAGNode>(op, index++);
        opToNode[op] = node.get();
        nodes.push_back(std::move(node));
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

    LLVM_DEBUG(llvm::dbgs() << "Built DAG with " << nodes.size() << " nodes\n");
  }

  // Find split points that minimize cross-segment dependencies
  std::vector<int> findSplitPoints(int splitCount) {
    if (nodes.empty() || splitCount <= 1) {
      return {};
    }

    int totalNodes = nodes.size();
    int nodesPerSegment = totalNodes / splitCount;

    std::vector<int> splitPoints;

    // Simple strategy: find points with minimal forward dependencies
    for (int i = 1; i < splitCount; i++) {
      int targetSplit = i * nodesPerSegment;
      int bestSplit = findBestSplitNear(targetSplit, nodesPerSegment / 2);
      splitPoints.push_back(bestSplit);
    }

    return splitPoints;
  }

 private:
  // Find the best split point near a target index
  int findBestSplitNear(int target, int searchRadius) {
    int bestSplit = target;
    int minCrossingEdges = INT_MAX;

    int start = std::max(1, target - searchRadius);
    int end = std::min((int)nodes.size() - 1, target + searchRadius);

    for (int split = start; split < end; split++) {
      int crossingEdges = countCrossingEdges(split);
      if (crossingEdges < minCrossingEdges) {
        minCrossingEdges = crossingEdges;
        bestSplit = split;
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "Best split at " << bestSplit << " with "
                            << minCrossingEdges << " crossing edges\n");
    return bestSplit;
  }

  // Count edges that cross a potential split point
  int countCrossingEdges(int splitPoint) {
    int crossingEdges = 0;

    for (int i = 0; i < splitPoint; i++) {
      for (DAGNode *successor : nodes[i]->successors) {
        if (successor->index >= splitPoint) {
          crossingEdges++;
        }
      }
    }

    return crossingEdges;
  }

 public:
  size_t getNodeCount() const { return nodes.size(); }
  DAGNode *getNode(int index) const {
    return index < nodes.size() ? nodes[index].get() : nullptr;
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

    // Check if function is already outlined (contains calls to segment
    // functions)
    bool alreadyOutlined = false;
    funcOp.walk(
        [&](func::CallOp callOp) {
          if (callOp.getCallee().str().find("_segment_") != std::string::npos) {
            alreadyOutlined = true;
            LLVM_DEBUG(
                llvm::dbgs()
                << "Function appears to be already outlined (contains call to "
                << callOp.getCallee() << ")\n");
          }
        });

    if (alreadyOutlined) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping already outlined function\n");
      funcOp.emitWarning() << "Function " << funcOp.getName()
                           << " appears to be already outlined or contains "
                              "invalid segment calls. "
                           << "Please use a fresh input file.";
      return;
    }

    // Analyze the function DAG
    SimpleDAGAnalyzer analyzer;
    analyzer.analyze(funcOp);

    LLVM_DEBUG(llvm::dbgs()
               << "Function " << funcOp.getName() << " has "
               << analyzer.getNodeCount() << " relevant operations\n");

    if (analyzer.getNodeCount() < static_cast<size_t>(splitCount) * 2) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Function too small to split (" << analyzer.getNodeCount()
                 << " nodes, need at least " << splitCount * 2 << ")\n");
      return;
    }

    // Find split points
    auto splitPoints = analyzer.findSplitPoints(splitCount);
    if (splitPoints.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No good split points found\n");
      return;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Found " << splitPoints.size() << " split points: ");
    for (size_t i = 0; i < splitPoints.size(); i++) {
      LLVM_DEBUG(llvm::dbgs() << splitPoints[i] << " ");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");

    // Perform the outlining
    performOutlining(funcOp, analyzer, splitPoints);
  }

 private:
  void performOutlining(func::FuncOp originalFunc, SimpleDAGAnalyzer &analyzer,
                        const std::vector<int> &splitPoints) {
    auto module = originalFunc->getParentOfType<ModuleOp>();
    OpBuilder builder(module.getContext());
    builder.setInsertionPointAfter(originalFunc);

    std::string baseName = originalFunc.getName().str();

    // Create segments based on split points
    std::vector<std::pair<int, int>> segments;  // (start, end) pairs
    int start = 0;
    for (int splitPoint : splitPoints) {
      segments.push_back({start, splitPoint});
      start = splitPoint;
    }
    segments.push_back({start, (int)analyzer.getNodeCount()});  // Last segment

    // Store created functions for the main dispatcher
    std::vector<func::FuncOp> segmentFunctions;

    // First pass: determine the actual types that flow between segments
    std::vector<Type> segmentInputTypes;
    std::vector<Type> segmentOutputTypes;

    for (size_t i = 0; i < segments.size(); i++) {
      int startIdx = segments[i].first;
      int endIdx = segments[i].second;

      // Input type
      if (i == 0) {
        // First segment uses original function input
        segmentInputTypes.push_back(originalFunc.getFunctionType().getInput(0));
      } else {
        // Use the output type of the previous segment
        segmentInputTypes.push_back(segmentOutputTypes.back());
      }

      // Output type: find the actual output of the last operation in this
      // segment
      Type outputType =
          originalFunc.getFunctionType().getResult(0);  // Default fallback
      for (int j = endIdx - 1; j >= startIdx; j--) {
        if (auto node = analyzer.getNode(j)) {
          if (!node->op->getResults().empty()) {
            outputType = node->op->getResult(0).getType();
            break;
          }
        }
      }
      segmentOutputTypes.push_back(outputType);

      LLVM_DEBUG(llvm::dbgs()
                 << "Segment " << i << " types: " << segmentInputTypes[i]
                 << " -> " << outputType << "\n");
    }

    // Second pass: create segment functions with correct types
    for (size_t i = 0; i < segments.size(); i++) {
      auto segmentFunc = createSegmentFunctionWithTypes(
          builder, originalFunc, analyzer, segments[i], static_cast<int>(i),
          baseName, segmentInputTypes[i], segmentOutputTypes[i]);
      if (segmentFunc) {
        segmentFunctions.push_back(segmentFunc);
        LLVM_DEBUG(llvm::dbgs() << "Created segment function: "
                                << segmentFunc.getName() << "\n");
      }
    }

    // Only create main dispatcher if we successfully created segment functions
    if (!segmentFunctions.empty()) {
      createMainFunction(builder, originalFunc, segmentFunctions, baseName);

      // Rename the original function to avoid conflicts
      originalFunc.setName(baseName + "_original");
    }
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

    if (startIdx >= endIdx) return nullptr;

    LLVM_DEBUG(llvm::dbgs()
               << "Creating segment " << segmentIndex << " with operations "
               << startIdx << " to " << endIdx << "\n");

    // Get operations to clone in this segment
    std::vector<Operation *> opsToClone;
    for (int i = startIdx; i < endIdx; i++) {
      if (auto node = analyzer.getNode(i)) {
        opsToClone.push_back(node->op);
      }
    }

    if (opsToClone.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No operations to clone in segment "
                              << segmentIndex << "\n");
      return nullptr;
    }

    // Save current insertion point
    auto savedInsertionPoint = builder.saveInsertionPoint();

    // Set insertion point to module level
    builder.setInsertionPointAfter(originalFunc);

    // Create function type with the determined input/output types
    auto segmentFuncType = builder.getFunctionType({inputType}, {outputType});

    // Create function at module level
    std::string segmentName =
        baseName + "_segment_" + std::to_string(segmentIndex);
    auto segmentFunc = builder.create<func::FuncOp>(
        originalFunc.getLoc(), segmentName, segmentFuncType);

    // Create function body
    Block *segmentBody = segmentFunc.addEntryBlock();
    builder.setInsertionPointToStart(segmentBody);

    // Use IRMapping for value mapping
    IRMapping valueMap;

    // Find the value that should be mapped to our function argument
    Value inputValue = segmentFunc.getArgument(0);

    // For the first segment, map original function arguments
    if (segmentIndex == 0) {
      for (auto [origArg, newArg] :
           llvm::zip(originalFunc.getArguments(), segmentFunc.getArguments())) {
        valueMap.map(origArg, newArg);
      }
    } else {
      // For subsequent segments, we need to find what value from previous
      // segments should be mapped to our input. For simplicity, map the first
      // operand we encounter that comes from outside our segment to our
      // function argument.
      bool foundMapping = false;
      for (Operation *op : opsToClone) {
        for (Value operand : op->getOperands()) {
          if (auto defOp = operand.getDefiningOp()) {
            // Check if this operand is defined before our segment starts
            bool isFromPreviousSegment = false;
            for (int j = 0; j < startIdx; j++) {
              if (auto prevNode = analyzer.getNode(j)) {
                if (prevNode->op == defOp) {
                  isFromPreviousSegment = true;
                  break;
                }
              }
            }
            if (isFromPreviousSegment) {
              valueMap.map(operand, inputValue);
              foundMapping = true;
              break;
            }
          }
        }
        if (foundMapping) break;
      }
    }

    // Clone operations in the segment
    for (Operation *op : opsToClone) {
      // Handle operands that come from outside this segment
      for (auto operand : op->getOperands()) {
        if (auto defOp = operand.getDefiningOp()) {
          if (valueMap.lookupOrNull(operand) == nullptr) {
            // If this operand is defined outside our segment, map it to
            // function argument
            valueMap.map(operand, inputValue);
          }
        }
      }

      // Clone the operation
      auto clonedOp = builder.clone(*op, valueMap);

      // Update value mapping with results
      for (auto [origResult, newResult] :
           llvm::zip(op->getResults(), clonedOp->getResults())) {
        valueMap.map(origResult, newResult);
      }
    }

    // Create return statement with the last computed value
    Value returnValue = inputValue;  // Default fallback

    // Find the actual output value (should be the last operation that produces
    // our output type)
    for (auto it = opsToClone.rbegin(); it != opsToClone.rend(); ++it) {
      Operation *op = *it;
      if (!op->getResults().empty()) {
        Value lastResult = op->getResult(0);
        Value mappedResult = valueMap.lookupOrNull(lastResult);
        if (mappedResult && mappedResult.getType() == outputType) {
          returnValue = mappedResult;
          break;
        }
      }
    }

    builder.create<func::ReturnOp>(originalFunc.getLoc(), returnValue);

    // Restore insertion point
    builder.restoreInsertionPoint(savedInsertionPoint);

    LLVM_DEBUG(llvm::dbgs() << "Successfully created segment function: "
                            << segmentName << " with input type: " << inputType
                            << " and output type: " << outputType << "\n");
    return segmentFunc;
  }

  void createMainFunction(OpBuilder &builder, func::FuncOp originalFunc,
                          const std::vector<func::FuncOp> &segmentFunctions,
                          const std::string &baseName) {
    // Save current insertion point
    auto savedInsertionPoint = builder.saveInsertionPoint();

    // Set insertion point to module level
    builder.setInsertionPointAfter(originalFunc);

    // Create main dispatcher function
    std::string mainName = baseName + "_main";
    auto mainFunc = builder.create<func::FuncOp>(
        originalFunc.getLoc(), mainName, originalFunc.getFunctionType());

    // Create function body
    Block *mainBody = mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(mainBody);

    // Chain calls to segment functions
    Value currentValue = mainFunc.getArgument(0);

    for (auto segmentFunc : segmentFunctions) {
      // Create call to segment function
      auto callOp = builder.create<func::CallOp>(
          originalFunc.getLoc(), segmentFunc.getFunctionType().getResults(),
          segmentFunc.getName(), ValueRange{currentValue});

      currentValue = callOp.getResult(0);
    }

    // Return final result
    builder.create<func::ReturnOp>(originalFunc.getLoc(), currentValue);

    // Restore insertion point
    builder.restoreInsertionPoint(savedInsertionPoint);

    LLVM_DEBUG(llvm::dbgs()
               << "Created main dispatcher function: " << mainName << "\n");
  }
};

}  // namespace heir
}  // namespace mlir
