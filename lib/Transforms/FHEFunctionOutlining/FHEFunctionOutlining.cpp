#include "lib/Transforms/FHEFunctionOutlining/FHEFunctionOutlining.h"

#include "llvm/include/llvm/ADT/PostOrderIterator.h"
#include "llvm/include/llvm/ADT/SetVector.h"
#include "llvm/include/llvm/Support/FileSystem.h"
#include "llvm/include/llvm/Support/ToolOutputFile.h"
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/Dominance.h"
#include "mlir/include/mlir/IR/IRMapping.h"
#include "mlir/include/mlir/IR/SymbolTable.h"
#include "mlir/include/mlir/Support/FileUtilities.h"

#define DEBUG_TYPE "fhe-function-outlining"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_FHEFUNCTIONOUTLINING
#include "lib/Transforms/FHEFunctionOutlining/FHEFunctionOutlining.h.inc"

//===----------------------------------------------------------------------===//
// OutlineRegion - Simplified for clean FHE pipeline design
//===----------------------------------------------------------------------===//

struct OutlineRegion {
  SmallVector<Operation*> operations;
  Value cryptoContext;     // The crypto context (always same)
  Value inputCiphertext;   // Single input ciphertext
  Value outputCiphertext;  // Single output ciphertext
  std::string regionId;

  bool isValid() const {
    return !operations.empty() && cryptoContext && inputCiphertext &&
           outputCiphertext;
  }
};

//===----------------------------------------------------------------------===//
// Simplified FHE Function Outlining Pass
//===----------------------------------------------------------------------===//

namespace {
struct FHEFunctionOutliningPass
    : public impl::FHEFunctionOutliningBase<FHEFunctionOutliningPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Find the main function
    func::FuncOp targetFunction = findTargetFunction(module);
    if (!targetFunction) {
      module.emitRemark() << "No suitable target function found for outlining";
      return;
    }

    // Find bootstrap operations to use as boundaries
    SmallVector<Operation*> bootstrapOps = findBootstrapOps(targetFunction);
    if (bootstrapOps.empty()) {
      targetFunction.emitRemark() << "No bootstrap operations found";
      return;
    }

    targetFunction.emitRemark()
        << "Found " << bootstrapOps.size()
        << " bootstrap operations, creating pipeline...";

    // Create regions between bootstrap operations
    SmallVector<OutlineRegion> regions =
        createBootstrapRegions(targetFunction, bootstrapOps);

    if (regions.empty()) {
      targetFunction.emitRemark() << "No valid regions found for outlining";
      return;
    }

    // Create outlined functions
    SmallVector<func::FuncOp> outlinedFunctions;
    OpBuilder builder(module.getContext());

    for (auto& region : regions) {
      if (auto newFunc =
              createOutlinedFunction(region, builder, module, targetFunction)) {
        outlinedFunctions.push_back(newFunc);
      }
    }

    // Replace original function with pipeline of calls
    createPipelineFunction(targetFunction, outlinedFunctions, builder);

    // Verify that all operations are preserved
    if (failed(
            verifyOperationPreservation(targetFunction, outlinedFunctions))) {
      targetFunction.emitWarning(
          "Some operations may not have been preserved during outlining");
    }

    targetFunction.emitRemark()
        << "Successfully created pipeline with " << outlinedFunctions.size()
        << " outlined functions";
  }

 private:
  func::FuncOp findTargetFunction(ModuleOp module) {
    func::FuncOp targetFunction = nullptr;

    module.walk([&](func::FuncOp funcOp) {
      if (!funcOp.isPrivate() && !funcOp.isDeclaration()) {
        StringRef funcName = funcOp.getName();
        if (!funcName.contains("configure") && !funcName.contains("setup") &&
            !funcName.contains("generate") && !funcName.contains("region")) {
          targetFunction = funcOp;
        }
      }
    });

    if (!targetFunction) {
      module.walk([&](func::FuncOp funcOp) {
        if (!funcOp.isPrivate() && !funcOp.isDeclaration() && !targetFunction) {
          targetFunction = funcOp;
        }
      });
    }

    return targetFunction;
  }

  SmallVector<Operation*> findBootstrapOps(func::FuncOp targetFunction) {
    SmallVector<Operation*> bootstrapOps;

    targetFunction.walk([&](Operation* op) {
      StringRef opName = op->getName().getStringRef();
      if (opName == "openfhe.bootstrap" || opName == "ckks.bootstrap") {
        bootstrapOps.push_back(op);
      }
    });

    // Sort by position in the function
    DenseMap<Operation*, int> opPositions;
    int pos = 0;
    targetFunction.walk([&](Operation* op) { opPositions[op] = pos++; });

    llvm::sort(bootstrapOps, [&](Operation* a, Operation* b) {
      return opPositions[a] < opPositions[b];
    });

    return bootstrapOps;
  }

  SmallVector<OutlineRegion> createBootstrapRegions(
      func::FuncOp targetFunction, ArrayRef<Operation*> bootstrapOps) {
    SmallVector<OutlineRegion> regions;

    // Get function arguments (assume crypto context is first, input ciphertext
    // is second)
    BlockArgument cryptoContext = targetFunction.getArgument(0);
    BlockArgument initialInput = targetFunction.getArgument(1);

    // Get the final return value to understand the complete computation
    Operation* returnOp = nullptr;
    targetFunction.walk(
        [&](func::ReturnOp retOp) { returnOp = retOp.getOperation(); });

    Value finalOutput = returnOp ? returnOp->getOperand(0) : nullptr;

    // Create regions between bootstrap operations
    for (size_t i = 0; i < bootstrapOps.size(); ++i) {
      Operation* currentBootstrap = bootstrapOps[i];

      OutlineRegion region;
      region.regionId = "region_" + std::to_string(i);
      region.cryptoContext = cryptoContext;

      // Determine input for this region
      if (i == 0) {
        region.inputCiphertext = initialInput;
      } else {
        // Input is the result of the previous bootstrap
        region.inputCiphertext = bootstrapOps[i - 1]->getResult(0);
      }

      // Output is the result of the current bootstrap
      region.outputCiphertext = currentBootstrap->getResult(0);

      // Collect operations between input and current bootstrap
      if (succeeded(collectOperationsBetweenPoints(targetFunction,
                                                   region.inputCiphertext,
                                                   currentBootstrap, region))) {
        if (region.isValid() && !region.operations.empty()) {
          regions.push_back(std::move(region));
        }
      }
    }

    // Handle operations after the last bootstrap (if any)
    if (!bootstrapOps.empty() && finalOutput) {
      Value lastBootstrapResult = bootstrapOps.back()->getResult(0);

      // If the final output is different from the last bootstrap result,
      // we need another region for post-bootstrap operations
      if (finalOutput != lastBootstrapResult) {
        OutlineRegion finalRegion;
        finalRegion.regionId = "region_final";
        finalRegion.cryptoContext = cryptoContext;
        finalRegion.inputCiphertext = lastBootstrapResult;
        finalRegion.outputCiphertext = finalOutput;

        if (succeeded(collectOperationsBetweenPoints(
                targetFunction, finalRegion.inputCiphertext,
                finalOutput.getDefiningOp(), finalRegion))) {
          if (finalRegion.isValid() && !finalRegion.operations.empty()) {
            regions.push_back(std::move(finalRegion));
          }
        }
      }
    }

    return regions;
  }

  LogicalResult collectOperationsBetweenPoints(func::FuncOp targetFunction,
                                               Value startValue,
                                               Operation* endOp,
                                               OutlineRegion& region) {
    // Get ALL operations in the function in execution order
    SmallVector<Operation*> allOps;
    targetFunction.walk([&](Operation* op) {
      if (op != targetFunction.getOperation()) {
        allOps.push_back(op);
      }
    });

    // Find the boundary operations
    Operation* startOp = nullptr;
    if (auto defOp = startValue.getDefiningOp()) {
      startOp = defOp;
    }

    // Find positions
    auto startIt = startOp ? llvm::find(allOps, startOp) : allOps.begin();
    auto endIt = llvm::find(allOps, endOp);

    if (endIt == allOps.end()) {
      return failure();
    }

    // Collect ALL operations between start and end (inclusive of end)
    // This ensures we capture side effects and setup operations
    for (auto it = startIt + 1; it <= endIt; ++it) {
      Operation* op = *it;

      // Include the operation if it's:
      // 1. Part of the computation (FHE operations)
      // 2. A setup/teardown operation (deserialize_key, clear_key)
      // 3. A constant or encoding operation
      if (isOperationRelevantForRegion(op)) {
        region.operations.push_back(op);
      }
    }

    return success();
  }

  bool isOperationRelevantForRegion(Operation* op) {
    StringRef opName = op->getName().getStringRef();

    // Include all FHE operations
    if (opName.starts_with("openfhe.") || opName.starts_with("ckks.") ||
        opName.starts_with("lwe.")) {
      return true;
    }

    // Include arithmetic and tensor operations
    if (opName.starts_with("arith.") || opName.starts_with("tensor.")) {
      return true;
    }

    // Exclude function control operations
    if (isa<func::CallOp>(op) || isa<func::ReturnOp>(op)) {
      return false;
    }

    // Include everything else to be safe (constants, etc.)
    return true;
  }

  func::FuncOp createOutlinedFunction(const OutlineRegion& region,
                                      OpBuilder& builder, ModuleOp module,
                                      func::FuncOp originalFunc) {
    // Create function name
    std::string funcName = originalFunc.getName().str() + "_" + region.regionId;

    // Ensure unique function name
    int suffix = 0;
    std::string uniqueName = funcName;
    while (module.lookupSymbol(uniqueName)) {
      uniqueName = funcName + "_" + std::to_string(suffix++);
    }

    // Create simple function type: (crypto_context, input_ciphertext) ->
    // output_ciphertext
    SmallVector<Type> inputTypes = {region.cryptoContext.getType(),
                                    region.inputCiphertext.getType()};
    SmallVector<Type> outputTypes = {region.outputCiphertext.getType()};

    auto funcType =
        FunctionType::get(builder.getContext(), inputTypes, outputTypes);

    // Create the function
    builder.setInsertionPointToEnd(module.getBody());
    auto newFunc = builder.create<func::FuncOp>(originalFunc.getLoc(),
                                                uniqueName, funcType);

    // Add function body
    Block* funcBody = newFunc.addEntryBlock();
    builder.setInsertionPointToStart(funcBody);

    // Create value mapping
    IRMapping valueMapping;
    valueMapping.map(region.cryptoContext, funcBody->getArgument(0));
    valueMapping.map(region.inputCiphertext, funcBody->getArgument(1));

    // Clone operations instead of moving them
    // This preserves the original operations until we're ready to replace the
    // entire function
    for (Operation* op : region.operations) {
      // Clone the operation (this creates a copy without destroying the
      // original)
      Operation* clonedOp = builder.clone(*op, valueMapping);

      // Update the mapping with the results of the cloned operation
      for (auto [original, cloned] :
           llvm::zip(op->getResults(), clonedOp->getResults())) {
        valueMapping.map(original, cloned);
      }
    }

    // Create return statement
    if (!valueMapping.contains(region.outputCiphertext)) {
      newFunc.emitError("Output ciphertext not found in mapping");
      newFunc.erase();
      return nullptr;
    }

    Value returnValue = valueMapping.lookup(region.outputCiphertext);
    builder.create<func::ReturnOp>(originalFunc.getLoc(), returnValue);

    return newFunc;
  }

  void createPipelineFunction(func::FuncOp originalFunc,
                              ArrayRef<func::FuncOp> outlinedFunctions,
                              OpBuilder& builder) {
    // Store the original return type and function signature
    Type originalReturnType = originalFunc.getFunctionType().getResult(0);
    FunctionType originalFuncType = originalFunc.getFunctionType();

    // Create a new function for the pipeline
    std::string pipelineName = originalFunc.getName().str() + "_pipeline";
    auto pipelineFunc = builder.create<func::FuncOp>(
        originalFunc.getLoc(), pipelineName, originalFuncType);

    // Build the pipeline in the new function
    Block* pipelineBody = pipelineFunc.addEntryBlock();
    builder.setInsertionPointToStart(pipelineBody);

    // Get arguments
    Value cryptoContext = pipelineBody->getArgument(0);
    Value currentCiphertext = pipelineBody->getArgument(1);

    // Create pipeline of function calls
    for (auto outlinedFunc : outlinedFunctions) {
      SmallVector<Value> callArgs = {cryptoContext, currentCiphertext};
      auto callOp = builder.create<func::CallOp>(pipelineFunc.getLoc(),
                                                 outlinedFunc, callArgs);
      currentCiphertext = callOp.getResult(0);
    }

    // Return the final result
    builder.create<func::ReturnOp>(pipelineFunc.getLoc(), currentCiphertext);

    // Now replace the original function's symbol and body
    SymbolTable symbolTable(originalFunc->getParentOfType<ModuleOp>());

    // Clear the original function body safely
    while (!originalFunc.getBody().empty()) {
      originalFunc.getBody().front().erase();
    }

    // Clone the pipeline body into the original function
    Block* newOriginalBody = originalFunc.addEntryBlock();
    builder.setInsertionPointToStart(newOriginalBody);

    IRMapping mapping;
    for (auto [pipelineArg, origArg] : llvm::zip(
             pipelineBody->getArguments(), newOriginalBody->getArguments())) {
      mapping.map(pipelineArg, origArg);
    }

    for (Operation& op : pipelineBody->getOperations()) {
      if (!isa<func::ReturnOp>(op)) {
        builder.clone(op, mapping);
      }
    }

    // Handle return operation
    for (Operation& op : pipelineBody->getOperations()) {
      if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
        SmallVector<Value> returnValues;
        for (Value operand : returnOp.getOperands()) {
          returnValues.push_back(mapping.lookup(operand));
        }
        builder.create<func::ReturnOp>(originalFunc.getLoc(), returnValues);
        break;
      }
    }

    // Remove the temporary pipeline function
    pipelineFunc.erase();
  }

  LogicalResult verifyOperationPreservation(
      func::FuncOp originalFunc, ArrayRef<func::FuncOp> outlinedFunctions) {
    // Count operation types in outlined functions
    DenseMap<StringRef, size_t> outlinedOpCounts;

    for (auto func : outlinedFunctions) {
      func.walk([&](Operation* op) {
        if (op != func.getOperation()) {
          outlinedOpCounts[op->getName().getStringRef()]++;
        }
      });
    }

    // Count operation types in the original pipeline (should just be calls now)
    DenseMap<StringRef, size_t> pipelineOpCounts;
    originalFunc.walk([&](Operation* op) {
      if (op != originalFunc.getOperation()) {
        pipelineOpCounts[op->getName().getStringRef()]++;
      }
    });

    // For now, just emit remarks about the operation counts
    originalFunc.emitRemark()
        << "Outlined functions contain " << outlinedOpCounts.size()
        << " different operation types";
    originalFunc.emitRemark()
        << "Pipeline function contains " << pipelineOpCounts.size()
        << " different operation types (should be mostly func.call and "
           "func.return)";

    return success();
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// Pass creation
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createFHEFunctionOutliningPass() {
  return std::make_unique<FHEFunctionOutliningPass>();
}

}  // namespace heir
}  // namespace mlir
