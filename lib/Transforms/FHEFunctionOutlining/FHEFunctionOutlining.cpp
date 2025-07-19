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

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_FHEFUNCTIONOUTLINING
#include "lib/Transforms/FHEFunctionOutlining/FHEFunctionOutlining.h.inc"

//===----------------------------------------------------------------------===//
// OutlineRegion - represents a region to extract
//===----------------------------------------------------------------------===//

struct OutlineRegion {
  SmallVector<Operation *> operations;
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
  std::string regionId;

  FunctionType getFunctionType(MLIRContext *context) const {
    SmallVector<Type> inputTypes, outputTypes;
    for (Value input : inputs) inputTypes.push_back(input.getType());
    for (Value output : outputs) outputTypes.push_back(output.getType());
    return FunctionType::get(context, inputTypes, outputTypes);
  }

  bool isValid() const {
    return !operations.empty() && inputs.size() < 20 && outputs.size() < 10;
  }
};

//===----------------------------------------------------------------------===//
// FHE Function Outlining Pass
//===----------------------------------------------------------------------===//

namespace {
struct FHEFunctionOutliningPass
    : public impl::FHEFunctionOutliningBase<FHEFunctionOutliningPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Find the main function (not helper functions)
    func::FuncOp targetFunction = nullptr;
    module.walk([&](func::FuncOp funcOp) {
      if (!funcOp.isPrivate() && !funcOp.isDeclaration()) {
        StringRef funcName = funcOp.getName();
        // Skip helper functions like configure/setup
        if (!funcName.contains("configure") && !funcName.contains("setup") &&
            !funcName.contains("generate")) {
          targetFunction = funcOp;
        }
      }
    });

    if (!targetFunction) {
      // Fallback - just take the first non-private function
      module.walk([&](func::FuncOp funcOp) {
        if (!funcOp.isPrivate() && !funcOp.isDeclaration() && !targetFunction) {
          targetFunction = funcOp;
        }
      });
    }

    if (!targetFunction) return;

    // Find bootstrap operations
    SmallVector<Operation *> bootstrapOps;
    targetFunction.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "openfhe.bootstrap") {
        bootstrapOps.push_back(op);
      }
    });

    if (bootstrapOps.empty()) {
      targetFunction.emitRemark() << "No bootstrap operations found";
      return;
    }

    targetFunction.emitRemark()
        << "Found " << bootstrapOps.size() << " bootstrap operations";

    OpBuilder builder(module.getContext());
    unsigned outlinedCount = 0;
    SmallVector<func::FuncOp> outlinedFunctions;

    // Create regions around each bootstrap
    for (auto [i, bootstrapOp] : llvm::enumerate(bootstrapOps)) {
      OutlineRegion region = createBootstrapRegion(bootstrapOp, i);

      if (region.isValid() && region.operations.size() >= 3) {
        func::FuncOp newFunc = extractFunction(region, builder, module);
        if (newFunc && replaceWithCall(region, newFunc, builder)) {
          outlinedFunctions.push_back(newFunc);
          outlinedCount++;
          targetFunction.emitRemark()
              << "Outlined bootstrap region " << i << " with "
              << region.operations.size() << " operations";
        }
      }
    }

    targetFunction.emitRemark()
        << "Successfully outlined " << outlinedCount << " regions";

    // Write outlined functions to separate files
    if (!outlinedFunctions.empty()) {
      writeOutlinedFunctions(outlinedFunctions, module);
    }
  }

 private:
  // Create region around a bootstrap operation
  OutlineRegion createBootstrapRegion(Operation *bootstrapOp,
                                      unsigned regionId) {
    OutlineRegion region;
    region.regionId = "bootstrap_region_" + std::to_string(regionId);

    DenseSet<Operation *> visited;
    region.operations.push_back(bootstrapOp);
    visited.insert(bootstrapOp);

    // Add operations that feed into bootstrap (1-2 levels back)
    addProducers(bootstrapOp, region.operations, visited, 2);

    // Add operations that consume bootstrap result (1-2 levels forward)
    addConsumers(bootstrapOp, region.operations, visited, 2);

    // Compute inputs and outputs
    computeInputsOutputs(region);

    bootstrapOp->emitRemark()
        << "Created region " << regionId << " with " << region.operations.size()
        << " operations, " << region.inputs.size() << " inputs, "
        << region.outputs.size() << " outputs";

    return region;
  }

  // Add producer operations (backwards)
  void addProducers(Operation *op, SmallVector<Operation *> &ops,
                    DenseSet<Operation *> &visited, unsigned levels) {
    if (levels == 0) return;

    for (Value operand : op->getOperands()) {
      if (auto defOp = operand.getDefiningOp()) {
        if (shouldInclude(defOp) && !visited.contains(defOp)) {
          ops.push_back(defOp);
          visited.insert(defOp);
          addProducers(defOp, ops, visited, levels - 1);
        }
      }
    }
  }

  // Add consumer operations (forwards)
  void addConsumers(Operation *op, SmallVector<Operation *> &ops,
                    DenseSet<Operation *> &visited, unsigned levels) {
    if (levels == 0) return;

    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (shouldInclude(user) && !visited.contains(user)) {
          ops.push_back(user);
          visited.insert(user);
          addConsumers(user, ops, visited, levels - 1);
        }
      }
    }
  }

  // Check if operation should be included in region
  bool shouldInclude(Operation *op) {
    if (op->hasTrait<OpTrait::IsTerminator>() ||
        isa<func::CallOp, func::ReturnOp>(op)) {
      return false;
    }

    StringRef opName = op->getName().getStringRef();
    return opName.starts_with("ckks.") || opName.starts_with("lwe.") ||
           opName.starts_with("arith.") || opName.starts_with("tensor.");
  }

  // Compute region inputs and outputs
  void computeInputsOutputs(OutlineRegion &region) {
    DenseSet<Operation *> regionOps(region.operations.begin(),
                                    region.operations.end());
    DenseSet<Value> inputSet, outputSet;

    // Find inputs - values used but not defined in region
    for (Operation *op : region.operations) {
      for (Value operand : op->getOperands()) {
        if (auto defOp = operand.getDefiningOp()) {
          if (!regionOps.contains(defOp)) {
            inputSet.insert(operand);
          }
        } else {
          inputSet.insert(operand);  // Block argument
        }
      }
    }

    // Find outputs - values defined in region but used outside
    for (Operation *op : region.operations) {
      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (!regionOps.contains(user)) {
            outputSet.insert(result);
            break;
          }
        }
      }
    }

    region.inputs.assign(inputSet.begin(), inputSet.end());
    region.outputs.assign(outputSet.begin(), outputSet.end());
  }

  // Extract region into new function
  func::FuncOp extractFunction(const OutlineRegion &region, OpBuilder &builder,
                               ModuleOp module) {
    builder.setInsertionPointToEnd(module.getBody());

    auto funcType = region.getFunctionType(builder.getContext());
    auto newFunc = builder.create<func::FuncOp>(builder.getUnknownLoc(),
                                                region.regionId, funcType);

    Block *funcBody = newFunc.addEntryBlock();
    builder.setInsertionPointToStart(funcBody);

    // Map inputs to function arguments
    IRMapping valueMapping;
    for (auto [input, arg] :
         llvm::zip(region.inputs, funcBody->getArguments())) {
      valueMapping.map(input, arg);
    }

    // Clone operations
    for (Operation *op : region.operations) {
      builder.clone(*op, valueMapping);
    }

    // Create return
    SmallVector<Value> returnValues;
    for (Value output : region.outputs) {
      returnValues.push_back(valueMapping.lookup(output));
    }
    builder.create<func::ReturnOp>(builder.getUnknownLoc(), returnValues);

    return newFunc;
  }

  // Replace region with function call
  bool replaceWithCall(const OutlineRegion &region, func::FuncOp newFunc,
                       OpBuilder &builder) {
    if (region.operations.empty()) return false;

    Operation *firstOp = region.operations[0];
    builder.setInsertionPoint(firstOp);

    auto callOp =
        builder.create<func::CallOp>(firstOp->getLoc(), newFunc, region.inputs);

    // Replace uses
    for (auto [output, result] :
         llvm::zip(region.outputs, callOp.getResults())) {
      Value mutableOutput = output;
      mutableOutput.replaceAllUsesWith(result);
    }

    // Erase operations in reverse order
    SmallVector<Operation *> opsToErase = region.operations;
    std::reverse(opsToErase.begin(), opsToErase.end());
    for (Operation *op : opsToErase) {
      op->erase();
    }

    return true;
  }

  // Write outlined functions to separate files
  void writeOutlinedFunctions(ArrayRef<func::FuncOp> functions,
                              ModuleOp originalModule) {
    MLIRContext *context = originalModule.getContext();

    for (auto [i, func] : llvm::enumerate(functions)) {
      // Create a new module for this function
      OpBuilder builder(context);
      auto newModule = builder.create<ModuleOp>(
          builder.getUnknownLoc(), "outlined_module_" + std::to_string(i));

      // Clone the function into the new module
      builder.setInsertionPointToEnd(newModule.getBody());
      func::FuncOp mutableFunc = func;
      builder.clone(*mutableFunc.getOperation());

      // Write to file
      std::string filename = "outlined_fhe_" + std::to_string(i) + ".mlir";
      std::error_code ec;
      llvm::raw_fd_ostream file(filename, ec);
      if (!ec) {
        newModule.print(file);
        file.close();
        originalModule.emitRemark()
            << "Wrote outlined function to " << filename;
      } else {
        originalModule.emitWarning()
            << "Failed to write " << filename << ": " << ec.message();
      }

      newModule.erase();
    }
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
