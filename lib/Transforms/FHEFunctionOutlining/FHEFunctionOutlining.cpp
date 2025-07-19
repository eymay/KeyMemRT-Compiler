#include "lib/Transforms/FHEFunctionOutlining/FHEFunctionOutlining.h"

#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheDialect.h"
#include "lib/Dialect/Openfhe/IR/OpenfheTypes.h"
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
#include "mlir/include/mlir/IR/Types.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Verifier.h"
#include "mlir/include/mlir/Support/FileUtilities.h"

#define DEBUG_TYPE "fhe-function-outlining"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_FHEFUNCTIONOUTLINING
#include "lib/Transforms/FHEFunctionOutlining/FHEFunctionOutlining.h.inc"

static constexpr StringRef kLinearTransformStartAttr =
    "heir.linear_transform_start";
static constexpr StringRef kLinearTransformEndAttr =
    "heir.linear_transform_end";

struct OutlineRegion {
  SmallVector<Operation *> operations;
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
  std::string regionId;
  int64_t linearTransformId = -1;
  Operation *startOp = nullptr;
  Operation *endOp = nullptr;

  FunctionType getFunctionType(MLIRContext *context) const {
    SmallVector<Type> inputTypes, outputTypes;
    for (Value input : inputs) inputTypes.push_back(input.getType());
    for (Value output : outputs) outputTypes.push_back(output.getType());
    return FunctionType::get(context, inputTypes, outputTypes);
  }

  bool isValid() const {
    return !operations.empty() && linearTransformId >= 0 && startOp && endOp;
  }
};

struct SafeRegionData {
  std::string funcName;
  SmallVector<Value> inputs;
  SmallVector<Value> outputs;
  SmallVector<Operation *> operations;
  Operation *insertPoint;
  int regionId;

  SafeRegionData(const OutlineRegion &region, StringRef originalFuncName) {
    funcName = ("outlined_" + region.regionId + "_" + originalFuncName).str();
    inputs.assign(region.inputs.begin(), region.inputs.end());
    outputs.assign(region.outputs.begin(), region.outputs.end());
    operations.assign(region.operations.begin(), region.operations.end());
    insertPoint = region.startOp;
    regionId = region.linearTransformId;
  }
};

struct FHEFunctionOutlining
    : public impl::FHEFunctionOutliningBase<FHEFunctionOutlining> {
  using FHEFunctionOutliningBase::FHEFunctionOutliningBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<func::FuncOp> funcsToProcess;
    module.walk([&](func::FuncOp func) { funcsToProcess.push_back(func); });

    for (func::FuncOp func : funcsToProcess) {
      processFunction(func, module);
    }
  }

 private:
  void processFunction(func::FuncOp func, ModuleOp module) {
    SmallVector<OutlineRegion> linearTransformRegions =
        collectLinearTransformRegions(func);

    if (linearTransformRegions.empty()) {
      llvm::outs() << "No linear transform regions found in function "
                   << func.getName() << " - skipping outlining\n";
      return;
    }

    llvm::outs() << "Found " << linearTransformRegions.size()
                 << " linear transform regions in function " << func.getName()
                 << "\n";

    processAllRegionsSafely(linearTransformRegions, func, module);
  }

  void processAllRegionsSafely(SmallVector<OutlineRegion> &regions,
                               func::FuncOp originalFunc, ModuleOp module) {
    SmallVector<SafeRegionData> safeData;

    for (auto &region : regions) {
      if (!shouldOutlineRegion(region)) continue;

      safeData.emplace_back(region, originalFunc.getName());
      llvm::outs() << "Prepared region " << region.linearTransformId
                   << " for outlining (" << region.operations.size()
                   << " ops)\n";
    }

    SmallVector<func::FuncOp> outlinedFuncs;
    OpBuilder builder(&getContext());

    for (auto &data : safeData) {
      func::FuncOp newFunc = createSafeOutlinedFunction(data, builder, module);
      outlinedFuncs.push_back(newFunc);
      llvm::outs() << "Created function " << data.funcName << "\n";
    }

    for (size_t i = 0; i < safeData.size(); ++i) {
      replaceSafely(safeData[i], outlinedFuncs[i]);
      llvm::outs() << "Successfully outlined region " << safeData[i].regionId
                   << "\n";
    }
  }

  SmallVector<OutlineRegion> collectLinearTransformRegions(func::FuncOp func) {
    SmallVector<OutlineRegion> regions;
    DenseMap<int64_t, OutlineRegion> regionMap;

    func.walk([&](Operation *op) {
      if (auto startAttr =
              op->getAttrOfType<IntegerAttr>(kLinearTransformStartAttr)) {
        int64_t regionId = startAttr.getInt();
        if (regionMap.find(regionId) == regionMap.end()) {
          regionMap[regionId] = OutlineRegion();
          regionMap[regionId].linearTransformId = regionId;
          regionMap[regionId].regionId =
              "linear_transform_" + std::to_string(regionId);
        }
        regionMap[regionId].startOp = op;

        llvm::outs() << "Found linear transform start marker on "
                     << op->getName() << " with id " << regionId << "\n";
      }

      if (auto endAttr =
              op->getAttrOfType<IntegerAttr>(kLinearTransformEndAttr)) {
        int64_t regionId = endAttr.getInt();
        if (regionMap.find(regionId) == regionMap.end()) {
          regionMap[regionId] = OutlineRegion();
          regionMap[regionId].linearTransformId = regionId;
          regionMap[regionId].regionId =
              "linear_transform_" + std::to_string(regionId);
        }
        regionMap[regionId].endOp = op;

        llvm::outs() << "Found linear transform end marker on " << op->getName()
                     << " with id " << regionId << "\n";
      }
    });

    for (auto &[regionId, region] : regionMap) {
      if (region.startOp && region.endOp) {
        collectOperationsBetween(region, func);
        computeInputsOutputs(region);
        regions.push_back(std::move(region));
      }
    }

    return regions;
  }

  void collectOperationsBetween(OutlineRegion &region, func::FuncOp func) {
    SmallVector<Operation *> allOps;
    func.walk([&](Operation *op) {
      if (op != func.getOperation()) {
        allOps.push_back(op);
      }
    });

    bool foundStart = false;
    for (Operation *op : allOps) {
      if (op == region.startOp) {
        foundStart = true;
        region.operations.push_back(op);
        continue;
      }

      if (foundStart) {
        if (shouldIncludeOperation(op)) {
          region.operations.push_back(op);
        }

        if (op == region.endOp) {
          if (region.operations.back() != op && shouldIncludeOperation(op)) {
            region.operations.push_back(op);
          }
          break;
        }
      }
    }

    llvm::outs() << "Collected " << region.operations.size()
                 << " operations for linear transform region "
                 << region.linearTransformId << "\n";
  }

  bool shouldIncludeOperation(Operation *op) {
    if (op->hasTrait<OpTrait::IsTerminator>() ||
        isa<func::CallOp, func::ReturnOp>(op)) {
      return false;
    }

    StringRef opName = op->getName().getStringRef();
    if (opName == "openfhe.deserialize_key" || opName == "openfhe.clear_key") {
      return false;
    }

    return opName.starts_with("openfhe.") || opName.starts_with("lwe.") ||
           opName.starts_with("arith.") || opName.starts_with("tensor.");
  }

  void computeInputsOutputs(OutlineRegion &region) {
    DenseSet<Operation *> regionOps(region.operations.begin(),
                                    region.operations.end());
    DenseSet<Value> inputSet;

    // Find inputs
    for (Operation *op : region.operations) {
      for (Value operand : op->getOperands()) {
        if (auto defOp = operand.getDefiningOp()) {
          if (!regionOps.contains(defOp)) {
            inputSet.insert(operand);
          }
        } else {
          inputSet.insert(operand);
        }
      }
    }

    // Debug: categorize and show the inputs
    llvm::outs() << "DEBUG: Analyzing inputs for region "
                 << region.linearTransformId << ":\n";
    int ccCount = 0, ctCount = 0, ptCount = 0, keyCount = 0, otherCount = 0;
    int debugCount = 0;

    for (Value input : inputSet) {
      StringRef typeName = input.getType().getDialect().getNamespace();
      StringRef typeStr = "unknown";

      if (typeName == "openfhe") {
        if (isa<openfhe::CryptoContextType>(input.getType())) {
          ccCount++;
          typeStr = "crypto_context";
        } else if (isa<openfhe::EvalKeyType>(input.getType())) {
          keyCount++;
          typeStr = "eval_key";
        } else {
          otherCount++;
          typeStr = "other_openfhe";
        }
      } else if (typeName == "lwe") {
        if (isa<lwe::NewLWECiphertextType>(input.getType())) {
          ctCount++;
          typeStr = "ciphertext";
        } else if (isa<lwe::NewLWEPlaintextType>(input.getType())) {
          ptCount++;
          typeStr = "plaintext";
        } else {
          otherCount++;
          typeStr = "other_lwe";
        }
      } else {
        otherCount++;
        typeStr = typeName;
      }

      if (debugCount < 15) {  // Show first 15 inputs
        llvm::outs() << "  Input[" << debugCount << "]: " << typeStr << "\n";
      }
      debugCount++;
    }

    llvm::outs() << "DEBUG: Input summary: " << ccCount << " crypto_context, "
                 << ctCount << " ciphertext, " << ptCount << " plaintext, "
                 << keyCount << " eval_keys, " << otherCount << " others\n";

    // Find the operation with linear_transform_end attribute for output
    SmallVector<Value> outputVec;
    Operation *endOp = nullptr;

    for (Operation *op : region.operations) {
      if (op->hasAttr(kLinearTransformEndAttr)) {
        endOp = op;
        if (op->getNumResults() > 0) {
          outputVec.push_back(op->getResult(0));
          llvm::outs() << "Found end operation " << op->getName()
                       << " with result as single output\n";
        }
        break;
      }
    }

    if (outputVec.empty()) {
      llvm::outs()
          << "WARNING: No linear_transform_end operation found in region "
          << region.linearTransformId << "\n";

      // Debug: show what intermediate values are used outside
      llvm::outs() << "DEBUG: Intermediate values used outside region "
                   << region.linearTransformId << ":\n";
      DenseSet<Value> externalUses;
      int debugCount2 = 0;

      for (Operation *op : region.operations) {
        for (Value result : op->getResults()) {
          for (Operation *user : result.getUsers()) {
            if (!regionOps.contains(user)) {
              externalUses.insert(result);
              if (debugCount2 < 10) {
                llvm::outs() << "  " << op->getName() << " result used by "
                             << user->getName() << " (outside region)\n";
              }
              debugCount2++;
              break;
            }
          }
        }
      }
      llvm::outs() << "DEBUG: Found " << externalUses.size()
                   << " values used externally\n";
    }

    region.inputs.assign(inputSet.begin(), inputSet.end());
    region.outputs = std::move(outputVec);

    llvm::outs() << "Region " << region.linearTransformId << " has "
                 << region.inputs.size() << " inputs and "
                 << region.outputs.size() << " outputs\n";
  }

  bool shouldOutlineRegion(const OutlineRegion &region) {
    if (region.operations.size() < 2) {
      llvm::outs() << "Skipping region " << region.linearTransformId
                   << " - too few operations\n";
      return false;
    }

    if (region.outputs.size() != 1) {
      llvm::outs() << "Skipping region " << region.linearTransformId
                   << " - expected 1 output, got " << region.outputs.size()
                   << "\n";
      return false;
    }

    if (region.inputs.size() > 2000) {
      llvm::outs() << "Skipping region " << region.linearTransformId
                   << " - too many inputs\n";
      return false;
    }

    llvm::outs() << "Region " << region.linearTransformId
                 << " dataflow: " << region.inputs.size() << " inputs, "
                 << region.outputs.size() << " outputs\n";

    return true;
  }

  func::FuncOp createSafeOutlinedFunction(const SafeRegionData &data,
                                          OpBuilder &builder, ModuleOp module) {
    builder.setInsertionPointToEnd(module.getBody());

    if (data.outputs.size() != 1) {
      throw std::runtime_error(
          "Linear transform region must have exactly 1 output");
    }

    SmallVector<Type> inputTypes;
    for (Value input : data.inputs) {
      if (!input || !input.getType()) {
        throw std::runtime_error("Invalid input value");
      }
      inputTypes.push_back(input.getType());
    }

    Type outputType = data.outputs[0].getType();
    if (!outputType) {
      throw std::runtime_error("Invalid output type");
    }

    auto funcType =
        FunctionType::get(builder.getContext(), inputTypes, {outputType});
    auto newFunc = builder.create<func::FuncOp>(builder.getUnknownLoc(),
                                                data.funcName, funcType);

    Block *newBlock = newFunc.addEntryBlock();
    builder.setInsertionPointToStart(newBlock);

    IRMapping valueMapping;
    for (unsigned i = 0; i < data.inputs.size(); ++i) {
      valueMapping.map(data.inputs[i], newBlock->getArgument(i));
    }

    for (Operation *op : data.operations) {
      if (!op) {
        throw std::runtime_error("Invalid operation pointer");
      }

      Operation *clonedOp = builder.clone(*op, valueMapping);
      clonedOp->removeAttr(kLinearTransformStartAttr);
      clonedOp->removeAttr(kLinearTransformEndAttr);
    }

    Value mappedOutput = valueMapping.lookup(data.outputs[0]);
    if (!mappedOutput) {
      throw std::runtime_error("Failed to find mapped output value");
    }

    builder.create<func::ReturnOp>(builder.getUnknownLoc(), mappedOutput);
    return newFunc;
  }

  void replaceSafely(const SafeRegionData &data, func::FuncOp newFunc) {
    OpBuilder builder(&getContext());

    if (!data.insertPoint) {
      throw std::runtime_error("Invalid insertion point");
    }

    builder.setInsertionPoint(data.insertPoint);

    for (Value input : data.inputs) {
      if (!input) {
        throw std::runtime_error("Null input value for function call");
      }
    }

    auto callOp = builder.create<func::CallOp>(
        builder.getUnknownLoc(), newFunc.getFunctionType().getResults(),
        newFunc.getName(), data.inputs);

    SmallVector<Value> outputs(data.outputs.begin(), data.outputs.end());
    for (unsigned i = 0; i < outputs.size(); ++i) {
      if (outputs[i]) {
        outputs[i].replaceAllUsesWith(callOp.getResult(i));
      }
    }

    for (Operation *op : llvm::reverse(data.operations)) {
      if (op) {
        op->erase();
      }
    }
  }
};

}  // namespace heir
}  // namespace mlir
