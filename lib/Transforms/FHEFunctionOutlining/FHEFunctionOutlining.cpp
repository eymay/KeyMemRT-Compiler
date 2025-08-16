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

// Simple DAG analyzer for finding split points
class SimpleDAGAnalyzer {
 private:
  std::vector<std::unique_ptr<DAGNode>> nodes;
  llvm::DenseMap<Operation *, DAGNode *> opToNode;

 public:
  void analyze(func::FuncOp funcOp) {
    // First pass: Create nodes for all relevant operations
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

  // Get constants that are directly used by operations in a segment
  llvm::DenseSet<Operation *> getSegmentConstants(int startIdx, int endIdx) {
    llvm::DenseSet<Operation *> segmentConstants;

    // Only include constants that are directly used by operations in this
    // segment
    for (int i = startIdx; i < endIdx; i++) {
      if (auto node = getNode(i)) {
        Operation *op = node->op;
        if (!isa<arith::ConstantOp>(op)) {
          // Check each operand of this operation
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

  // Find split points that minimize cross-segment dependencies
  std::vector<int> findSplitPoints(int splitCount) {
    if (nodes.empty() || splitCount <= 1) {
      return {};
    }

    int totalNodes = nodes.size();
    int nodesPerSegment = totalNodes / splitCount;

    std::vector<int> splitPoints;

    // Simple strategy: divide roughly equally by node count
    // but try to avoid splitting in the middle of heavy dependency chains
    for (int i = 1; i < splitCount; i++) {
      int candidateSplit = i * nodesPerSegment;

      // Look for a better split point within a small window
      int bestSplit = candidateSplit;
      int minDependencies = INT_MAX;

      int windowStart = std::max(0, candidateSplit - 2);
      int windowEnd = std::min(totalNodes, candidateSplit + 3);

      for (int j = windowStart; j < windowEnd; j++) {
        int crossDeps = countCrossDependencies(j);
        if (crossDeps < minDependencies) {
          minDependencies = crossDeps;
          bestSplit = j;
        }
      }

      splitPoints.push_back(bestSplit);
    }

    LLVM_DEBUG(llvm::dbgs() << "Found split points: ");
    for (int sp : splitPoints) {
      LLVM_DEBUG(llvm::dbgs() << sp << " ");
    }
    LLVM_DEBUG(llvm::dbgs() << "\n");

    return splitPoints;
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
    return (index >= 0 && index < static_cast<int>(nodes.size()))
               ? nodes[index].get()
               : nullptr;
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
    segments.emplace_back(start, analyzer.size());  // Last segment

    LLVM_DEBUG(llvm::dbgs() << "Created " << segments.size() << " segments\n");

    // Get function types for segments with proper type propagation
    auto [segmentInputTypes, segmentOutputTypes] =
        determineSegmentTypes(funcOp, analyzer, segments);

    // Create outlined functions
    OpBuilder builder(&getContext());
    std::string baseName = funcOp.getName().str();
    std::vector<func::FuncOp> segmentFunctions;

    for (size_t i = 0; i < segments.size(); i++) {
      auto segmentFunc = createSegmentFunctionWithTypes(
          builder, funcOp, analyzer, segments[i], static_cast<int>(i), baseName,
          segmentInputTypes[i], segmentOutputTypes[i]);
      if (segmentFunc) {
        segmentFunctions.push_back(segmentFunc);
        LLVM_DEBUG(llvm::dbgs() << "Created segment function: "
                                << segmentFunc.getName() << "\n");
      }
    }

    // Only create main dispatcher if we successfully created segment functions
    if (!segmentFunctions.empty()) {
      createMainFunction(builder, funcOp, segmentFunctions, baseName);

      // Instead of clearing the function body (which causes MLIR validation
      // issues), just rename it to indicate it's the old version. A later
      // cleanup pass can remove it entirely if needed.
      funcOp.setPrivate();
      funcOp.setName(baseName + "_original");
    }
  }

  std::vector<Type> determineSegmentInputTypes(
      func::FuncOp funcOp, SimpleDAGAnalyzer &analyzer,
      const std::vector<std::pair<int, int>> &segments) {
    std::vector<Type> inputTypes;

    for (size_t i = 0; i < segments.size(); i++) {
      Type inputType;

      if (i == 0) {
        // First segment: input type is the original function's first argument
        inputType = funcOp.getArgumentTypes()[0];
      } else {
        // Later segments: we'll determine this after we know the output types
        // For now, use a placeholder - we'll fix this in
        // determineSegmentOutputTypes
        inputType = funcOp.getArgumentTypes()[0];
      }

      inputTypes.push_back(inputType);
    }

    return inputTypes;
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

      // Find the last computational operation in this segment and use its
      // result type
      for (int j = segment.second - 1; j >= segment.first; j--) {
        if (auto node = analyzer.getNode(j)) {
          if (!isa<arith::ConstantOp>(node->op) &&
              !node->op->getResults().empty()) {
            outputType = node->op->getResult(0).getType();
            llvm::dbgs() << "  Found last operation at index " << j << ": "
                         << *node->op << "\n";
            llvm::dbgs() << "  Operation result type: " << outputType << "\n";

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

    if (startIdx >= endIdx) return nullptr;

    LLVM_DEBUG(llvm::dbgs()
               << "Creating segment " << segmentIndex << " with operations "
               << startIdx << " to " << endIdx << "\n");

    // Get all operations for this segment in their original order
    std::vector<Operation *> allOpsToClone;

    // Walk the original function in order and collect operations that belong to
    // this segment
    auto segmentConstants = analyzer.getSegmentConstants(startIdx, endIdx);

    originalFunc.walk<WalkOrder::PreOrder>([&](Operation *op) {
      // Skip the function operation itself
      if (op == originalFunc.getOperation()) {
        return WalkResult::advance();
      }

      // Check if this is a constant needed by this segment
      if (isa<arith::ConstantOp>(op) && segmentConstants.contains(op)) {
        allOpsToClone.push_back(op);
        return WalkResult::advance();
      }

      // Check if this is a computational operation in this segment
      for (int i = startIdx; i < endIdx; i++) {
        if (auto node = analyzer.getNode(i)) {
          if (node->op == op && !isa<arith::ConstantOp>(op)) {
            allOpsToClone.push_back(op);
            break;
          }
        }
      }

      return WalkResult::advance();
    });

    if (allOpsToClone.empty()) {
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
      for (Operation *op : allOpsToClone) {
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

    // Clone operations in original order
    for (Operation *op : allOpsToClone) {
      // Handle operands that come from outside this segment
      for (auto operand : op->getOperands()) {
        if (auto defOp = operand.getDefiningOp()) {
          if (valueMap.lookupOrNull(operand) == nullptr) {
            // If this operand is defined outside our segment, map it to
            // function argument (we don't need to check segmentConstants here
            // since we're processing in order)
            bool isFromPreviousSegment = true;
            for (Operation *segmentOp : allOpsToClone) {
              if (segmentOp == defOp) {
                isFromPreviousSegment = false;
                break;
              }
            }
            if (isFromPreviousSegment) {
              valueMap.map(operand, inputValue);
            }
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

      if (isa<arith::ConstantOp>(op)) {
        LLVM_DEBUG(llvm::dbgs() << "Cloned constant operation in segment "
                                << segmentIndex << "\n");
      }
    }

    // Find the actual output value (should be the last computational operation
    // that produces our output type)
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

    // Restore insertion point
    builder.restoreInsertionPoint(savedInsertionPoint);

    LLVM_DEBUG(llvm::dbgs() << "Successfully created segment function: "
                            << segmentName << " with input type: " << inputType
                            << " and output type: " << outputType << " and "
                            << segmentConstants.size() << " constants\n");
    return segmentFunc;
  }

  void createMainFunction(OpBuilder &builder, func::FuncOp originalFunc,
                          const std::vector<func::FuncOp> &segmentFunctions,
                          const std::string &baseName) {
    // Save current insertion point
    auto savedInsertionPoint = builder.saveInsertionPoint();

    // Set insertion point to module level
    builder.setInsertionPointAfter(originalFunc);

    // Create main function with the same signature as original
    std::string mainFuncName = baseName + "_main";
    auto mainFuncType = originalFunc.getFunctionType();
    auto mainFunc = builder.create<func::FuncOp>(originalFunc.getLoc(),
                                                 mainFuncName, mainFuncType);

    // Create function body
    Block *mainBody = mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(mainBody);

    // Chain calls to segment functions
    Value currentValue = mainFunc.getArgument(0);

    for (auto segmentFunc : segmentFunctions) {
      // Call the segment function
      auto callOp = builder.create<func::CallOp>(
          originalFunc.getLoc(), segmentFunc.getName(),
          segmentFunc.getResultTypes(), ValueRange{currentValue});
      currentValue = callOp.getResult(0);
    }

    // Return the final result
    builder.create<func::ReturnOp>(originalFunc.getLoc(), currentValue);

    // Restore insertion point
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
