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
  SmallVector<Operation *>
      internalConstants;  // Constants to be moved into outlined function
  SmallVector<Operation *>
      internalPlaintexts;  // Plaintext ops to be moved into outlined function
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
};

class FHEFunctionOutliningPass
    : public impl::FHEFunctionOutliningBase<FHEFunctionOutliningPass> {
 public:
  using FHEFunctionOutliningBase::FHEFunctionOutliningBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> functionsToProcess;

    module.walk([&](func::FuncOp func) {
      if (!func.isPrivate() && func.getName() != "main") {
        functionsToProcess.push_back(func);
      }
    });

    for (func::FuncOp func : functionsToProcess) {
      processFunction(func, module);
    }
  }

 private:
  // Track operations that have been erased to avoid double-free
  llvm::DenseSet<Operation *> erasedOperations;
  // Check if an operation can be moved into the outlined function
  bool isMovableToOutlinedFunction(Operation *op) {
    // Check if operation is valid first
    if (!op) {
      return false;
    }

    // Constants can always be copied
    if (isa<arith::ConstantOp>(op)) {
      return true;
    }

    StringRef opName = op->getName().getStringRef();

    // Plaintext creation operations can be moved
    if (opName == "openfhe.make_ckks_packed_plaintext" ||
        opName == "openfhe.make_packed_plaintext") {
      return true;
    }

    // Tensor operations that create constants
    if (opName.starts_with("tensor.") && op->getNumOperands() == 0) {
      return true;
    }

    return false;
  }

  // Check if a value should be treated as external input
  bool shouldBeExternalInput(Value value, const OutlineRegion &region) {
    Operation *definingOp = value.getDefiningOp();

    if (!definingOp) {
      // Block arguments (like function arguments) are external inputs
      return true;
    }

    // If the defining op is in our region's operations, it's not external
    if (llvm::find(region.operations, definingOp) != region.operations.end()) {
      return false;
    }

    // If it's a movable operation, it's not external (will be moved)
    if (isMovableToOutlinedFunction(definingOp)) {
      return false;
    }

    // Check if it's a crypto context or ciphertext that we should keep as input
    Type valueType = value.getType();

    // Crypto context should be external input
    if (isa<openfhe::CryptoContextType>(valueType)) {
      return true;
    }

    // Ciphertext values should be external input
    if (isa<lwe::NewLWECiphertextType>(valueType)) {
      return true;
    }

    // Evaluation keys should be external input
    if (isa<openfhe::EvalKeyType>(valueType)) {
      return true;
    }

    // Everything else (including plaintexts and constants) should not be
    // external
    return false;
  }

  void processFunction(func::FuncOp func, ModuleOp module) {
    // Clear the erased operations set for each function
    erasedOperations.clear();

    llvm::outs() << "Starting processFunction for: " << func.getName() << "\n";

    std::vector<OutlineRegion> regions;
    try {
      regions = collectLinearTransformRegions(func);
    } catch (const std::exception &e) {
      llvm::outs() << "ERROR in collectLinearTransformRegions: " << e.what()
                   << "\n";
      return;
    }

    llvm::outs() << "Found " << regions.size()
                 << " linear transform regions in function " << func.getName()
                 << "\n";

    if (regions.empty()) return;

    // PHASE 1: Make each region self-contained by copying shared operations
    try {
      llvm::outs() << "Starting PHASE 1: localization\n";
      localizeSharedOperations(regions, func);
      llvm::outs() << "Completed PHASE 1: localization\n";
    } catch (const std::exception &e) {
      llvm::outs() << "ERROR in PHASE 1: " << e.what() << "\n";
      return;
    }

    // PHASE 2: Process regions in reverse order for safe outlining
    std::sort(regions.begin(), regions.end(),
              [](const OutlineRegion &a, const OutlineRegion &b) {
                return a.linearTransformId > b.linearTransformId;
              });

    // PHASE 3: Recompute inputs/outputs after localization
    try {
      llvm::outs() << "Starting PHASE 3: recomputing inputs/outputs\n";
      for (auto &region : regions) {
        computeInputsOutputs(region);
      }
      llvm::outs() << "Completed PHASE 3: recomputing inputs/outputs\n";
    } catch (const std::exception &e) {
      llvm::outs() << "ERROR in PHASE 3: " << e.what() << "\n";
      return;
    }

    try {
      outlineRegions(regions, func, module);
    } catch (const std::exception &e) {
      llvm::outs() << "ERROR in outlineRegions: " << e.what() << "\n";
      return;
    }
  }

 private:
  // Phase 1: Localize shared operations to make regions self-contained
  void localizeSharedOperations(std::vector<OutlineRegion> &regions,
                                func::FuncOp func) {
    llvm::outs() << "PHASE 1: Localizing shared operations...\n";

    // Find all shared operations (constants and plaintexts used by multiple
    // regions)
    llvm::DenseMap<Operation *, std::vector<int64_t>>
        sharedOperations;  // op -> list of region IDs that use it

    for (auto &region : regions) {
      llvm::DenseSet<Operation *> regionOpsSet(region.operations.begin(),
                                               region.operations.end());

      for (Operation *op : region.operations) {
        for (Value operand : op->getOperands()) {
          Operation *definingOp = operand.getDefiningOp();
          if (definingOp && isMovableToOutlinedFunction(definingOp)) {
            // Check if this operation is outside our region
            if (regionOpsSet.find(definingOp) == regionOpsSet.end()) {
              sharedOperations[definingOp].push_back(region.linearTransformId);
            }
          }
        }
      }
    }

    // Create local copies for each shared operation
    OpBuilder builder(func.getContext());

    for (auto &[sharedOp, regionIds] : sharedOperations) {
      if (regionIds.size() >=
          1) {  // Even single-use operations need to be localized
        llvm::outs() << "  Localizing operation " << sharedOp->getName()
                     << " used by " << regionIds.size() << " regions: ";
        for (int64_t id : regionIds) llvm::outs() << id << " ";
        llvm::outs() << "\n";

        // Create a local copy for each region that uses this operation
        for (int64_t regionId : regionIds) {
          // Find the region
          auto regionIt = std::find_if(regions.begin(), regions.end(),
                                       [regionId](const OutlineRegion &r) {
                                         return r.linearTransformId == regionId;
                                       });
          if (regionIt != regions.end()) {
            createLocalCopy(sharedOp, *regionIt, builder);
          }
        }
      }
    }

    llvm::outs()
        << "PHASE 1 completed: Created local copies for shared operations\n";
  }

  void createLocalCopy(Operation *sharedOp, OutlineRegion &region,
                       OpBuilder &builder) {
    // Insert the copy at the beginning of the region
    if (region.operations.empty()) {
      return;  // No operations in region
    }

    builder.setInsertionPoint(region.operations[0]);

    llvm::outs() << "    Creating local copy of " << sharedOp->getName()
                 << " for region " << region.linearTransformId << "\n";

    IRMapping localMapping;

    // Handle the simple case: make_ckks_packed_plaintext + its arith.constant
    if (sharedOp->getName().getStringRef() ==
        "openfhe.make_ckks_packed_plaintext") {
      // First operand is crypto context (external), second is the constant
      Value constantOperand = sharedOp->getOperand(1);
      Operation *constantOp = constantOperand.getDefiningOp();

      if (constantOp && isa<arith::ConstantOp>(constantOp)) {
        // Clone the constant first
        Operation *constantCopy = builder.clone(*constantOp, localMapping);
        region.operations.insert(region.operations.begin(), constantCopy);
        llvm::outs() << "      Cloned constant dependency\n";
      }
    } else if (isa<arith::ConstantOp>(sharedOp)) {
      // For standalone constants, no dependencies needed
      llvm::outs() << "      Cloning standalone constant\n";
    }

    // Now clone the shared operation itself
    Operation *localCopy = builder.clone(*sharedOp, localMapping);

    // IMPORTANT: If this is the first operation we're adding to the region,
    // it should get the start marker. If it's a plaintext op, it might need the
    // start marker.
    if (sharedOp->getName().getStringRef() ==
        "openfhe.make_ckks_packed_plaintext") {
      // Check if this should be the start of the linear transform region
      bool shouldBeStart = false;
      for (Operation *regionOp : region.operations) {
        if (regionOp == region.startOp) {
          // If the start marker was on an operation that will be in the region,
          // we need to ensure our new plaintext op is properly marked
          shouldBeStart = true;
          break;
        }
      }

      if (shouldBeStart || region.operations.empty()) {
        // Add start marker to the local copy
        localCopy->setAttr(kLinearTransformStartAttr,
                           builder.getI64IntegerAttr(region.linearTransformId));
        llvm::outs() << "      Added start marker to localized plaintext\n";
      }
    }

    // Update ALL uses within this region to use the local copy
    Value originalResult = sharedOp->getResult(0);
    Value localResult = localCopy->getResult(0);

    SmallVector<std::pair<Operation *, unsigned>> usesToUpdate;

    // Collect all uses within this region
    for (Operation *user : originalResult.getUsers()) {
      if (llvm::find(region.operations, user) != region.operations.end()) {
        // Find which operand uses the original result
        for (unsigned i = 0; i < user->getNumOperands(); ++i) {
          if (user->getOperand(i) == originalResult) {
            usesToUpdate.push_back({user, i});
          }
        }
      }
    }

    // Update all the uses
    for (auto [user, operandIdx] : usesToUpdate) {
      user->setOperand(operandIdx, localResult);
      llvm::outs() << "      Updated use in " << user->getName() << "\n";
    }

    // Add the local copy to the region's operations at the beginning
    region.operations.insert(region.operations.begin(), localCopy);

    // Update the region's start operation if needed
    if (sharedOp == region.startOp ||
        (region.operations.size() == 1 &&
         sharedOp->getName().getStringRef() ==
             "openfhe.make_ckks_packed_plaintext")) {
      region.startOp = localCopy;
      llvm::outs()
          << "      Updated region start operation to localized copy\n";
    }

    llvm::outs() << "      Successfully localized " << sharedOp->getName()
                 << " for region " << region.linearTransformId << " (updated "
                 << usesToUpdate.size() << " uses)\n";
  }

  void computeInputsOutputs(OutlineRegion &region) {
    llvm::SetVector<Value> candidateInputs;
    llvm::SetVector<Value> outputs;

    llvm::outs() << "DEBUG: Computing inputs/outputs for region "
                 << region.linearTransformId << " (after localization)\n";

    // Collect all values used by operations in the region
    llvm::DenseSet<Operation *> regionOpsSet(region.operations.begin(),
                                             region.operations.end());

    for (Operation *op : region.operations) {
      for (Value operand : op->getOperands()) {
        Operation *definingOp = operand.getDefiningOp();

        // If the defining operation is not in our region, it's an external
        // input
        if (definingOp && regionOpsSet.find(definingOp) == regionOpsSet.end()) {
          if (shouldBeExternalInput(operand, region)) {
            candidateInputs.insert(operand);
          }
        } else if (!definingOp) {
          // This is a block argument (function parameter), should be external
          // input
          candidateInputs.insert(operand);
        }
      }
    }

    // Find outputs (results used outside the region)
    for (Operation *op : region.operations) {
      if (op) {
        for (Value result : op->getResults()) {
          bool hasExternalUse = false;

          for (Operation *user : result.getUsers()) {
            if (regionOpsSet.find(user) == regionOpsSet.end()) {
              hasExternalUse = true;
              break;
            }
          }

          if (hasExternalUse) {
            outputs.insert(result);
            llvm::outs() << "DEBUG: Found output from: " << op->getName()
                         << "\n";
          }
        }
      }
    }

    // Store the results
    region.inputs.assign(candidateInputs.begin(), candidateInputs.end());
    region.outputs.assign(outputs.begin(), outputs.end());

    // After localization, we shouldn't need movable constants/plaintexts
    region.internalConstants.clear();
    region.internalPlaintexts.clear();

    llvm::outs() << "DEBUG: Analyzing inputs for region "
                 << region.linearTransformId << ":\n";
    for (size_t i = 0; i < std::min(region.inputs.size(), size_t(15)); ++i) {
      Value input = region.inputs[i];
      Type inputType = input.getType();

      if (isa<lwe::NewLWEPlaintextType>(inputType)) {
        llvm::outs() << "  Input[" << i << "]: plaintext\n";
      } else if (isa<lwe::NewLWECiphertextType>(inputType)) {
        llvm::outs() << "  Input[" << i << "]: ciphertext\n";
      } else if (isa<openfhe::CryptoContextType>(inputType)) {
        llvm::outs() << "  Input[" << i << "]: crypto_context\n";
      } else if (isa<openfhe::EvalKeyType>(inputType)) {
        llvm::outs() << "  Input[" << i << "]: eval_keys\n";
      } else {
        llvm::outs() << "  Input[" << i << "]: builtin\n";
      }
    }

    // Count input types for summary
    int cryptoContext = 0, ciphertext = 0, plaintext = 0, evalKeys = 0,
        others = 0;
    for (Value input : region.inputs) {
      Type inputType = input.getType();
      if (isa<openfhe::CryptoContextType>(inputType))
        cryptoContext++;
      else if (isa<lwe::NewLWECiphertextType>(inputType))
        ciphertext++;
      else if (isa<lwe::NewLWEPlaintextType>(inputType))
        plaintext++;
      else if (isa<openfhe::EvalKeyType>(inputType))
        evalKeys++;
      else
        others++;
    }

    llvm::outs() << "DEBUG: Input summary: " << cryptoContext
                 << " crypto_context, " << ciphertext << " ciphertext, "
                 << plaintext << " plaintext, " << evalKeys << " eval_keys, "
                 << others << " others\n";
    llvm::outs() << "DEBUG: Movable constants: "
                 << region.internalConstants.size() << "\n";
    llvm::outs() << "DEBUG: Movable plaintexts: "
                 << region.internalPlaintexts.size() << "\n";

    // Debug: Show some examples of movable constants
    if (!region.internalConstants.empty()) {
      llvm::outs() << "DEBUG: Sample movable constants:\n";
      for (size_t i = 0;
           i < std::min(region.internalConstants.size(), size_t(5)); ++i) {
        Operation *op = region.internalConstants[i];
        if (op) {
          llvm::outs() << "  - " << op->getName()
                       << " (operands: " << op->getNumOperands() << ")\n";
        }
      }
    }
  }

  bool shouldOutlineRegion(const OutlineRegion &region) {
    if (region.operations.size() < 50) {
      llvm::outs() << "Skipping region " << region.linearTransformId
                   << " - too few operations (" << region.operations.size()
                   << ")\n";
      return false;
    }

    llvm::outs() << "DEBUG: Region " << region.linearTransformId << " has "
                 << region.outputs.size() << " outputs\n";

    if (region.outputs.size() != 1) {
      llvm::outs() << "Skipping region " << region.linearTransformId
                   << " - multiple outputs not supported (found "
                   << region.outputs.size() << " outputs)\n";

      // Debug: show what the outputs are
      for (size_t i = 0; i < region.outputs.size(); ++i) {
        Value output = region.outputs[i];
        Operation *definingOp = output.getDefiningOp();
        llvm::outs() << "  Output " << i << ": from "
                     << (definingOp ? definingOp->getName().getStringRef().str()
                                    : "block argument")
                     << "\n";
      }
      return false;
    }

    return true;
  }

  struct SafeRegionData {
    SmallVector<Operation *> operations;
    SmallVector<Value> inputs;
    SmallVector<Value> outputs;
    SmallVector<Operation *> internalConstants;
    SmallVector<Operation *> internalPlaintexts;
    std::string funcName;
    std::string regionId;
    Operation *insertPoint;

    SafeRegionData(const OutlineRegion &region, StringRef originalFuncName)
        : operations(region.operations),
          inputs(region.inputs),
          outputs(region.outputs),
          internalConstants(region.internalConstants),
          internalPlaintexts(region.internalPlaintexts),
          regionId(region.regionId) {
      funcName = ("outlined_" + region.regionId + "_" + originalFuncName).str();
      insertPoint = region.startOp;
    }
  };

  func::FuncOp createSafeOutlinedFunction(const SafeRegionData &data,
                                          OpBuilder &builder, ModuleOp module) {
    builder.setInsertionPointToStart(module.getBody());

    // Ensure crypto context is always the first argument
    SmallVector<Type> inputTypes;
    SmallVector<Value> orderedInputs;

    // First argument: crypto context
    Value cryptoContext = nullptr;
    for (Value input : data.inputs) {
      if (isa<openfhe::CryptoContextType>(input.getType())) {
        cryptoContext = input;
        break;
      }
    }

    if (cryptoContext) {
      inputTypes.push_back(cryptoContext.getType());
      orderedInputs.push_back(cryptoContext);
    }

    // Second and subsequent arguments: everything else (ciphertext, eval keys,
    // etc.)
    for (Value input : data.inputs) {
      if (input != cryptoContext) {
        inputTypes.push_back(input.getType());
        orderedInputs.push_back(input);
      }
    }

    FunctionType funcType = FunctionType::get(
        builder.getContext(), inputTypes,
        llvm::map_to_vector(data.outputs, [](Value v) { return v.getType(); }));

    auto newFunc = builder.create<func::FuncOp>(builder.getUnknownLoc(),
                                                data.funcName, funcType);

    Block *newBlock = newFunc.addEntryBlock();
    builder.setInsertionPointToStart(newBlock);

    IRMapping valueMapping;

    // Map ordered inputs to function arguments
    for (unsigned i = 0; i < orderedInputs.size(); ++i) {
      valueMapping.map(orderedInputs[i], newBlock->getArgument(i));
    }

    // Clone all operations (they should all be self-contained now)
    for (Operation *op : data.operations) {
      if (op) {
        Operation *clonedOp = builder.clone(*op, valueMapping);
        clonedOp->removeAttr(kLinearTransformStartAttr);
        clonedOp->removeAttr(kLinearTransformEndAttr);
      }
    }

    // Create return statement
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

    // Create ordered inputs matching the function signature
    SmallVector<Value> callInputs;

    // First argument: crypto context
    Value cryptoContext = nullptr;
    for (Value input : data.inputs) {
      if (isa<openfhe::CryptoContextType>(input.getType())) {
        cryptoContext = input;
        break;
      }
    }

    if (cryptoContext) {
      callInputs.push_back(cryptoContext);
    } else {
      // If no crypto context found in inputs, get it from movable plaintext
      // operations
      for (Operation *ptOp : data.internalPlaintexts) {
        if (ptOp && ptOp->getName().getStringRef() ==
                        "openfhe.make_ckks_packed_plaintext") {
          Value ccOperand =
              ptOp->getOperand(0);  // First operand is crypto context
          if (isa<openfhe::CryptoContextType>(ccOperand.getType())) {
            cryptoContext = ccOperand;
            callInputs.push_back(ccOperand);
            break;
          }
        }
      }
    }

    // Remaining arguments: everything else except crypto context
    for (Value input : data.inputs) {
      if (input != cryptoContext) {
        callInputs.push_back(input);
      }
    }

    // Validate inputs
    for (Value input : callInputs) {
      if (!input) {
        throw std::runtime_error("Null input value for function call");
      }
    }

    // Create function call
    auto callOp = builder.create<func::CallOp>(
        builder.getUnknownLoc(), newFunc.getFunctionType().getResults(),
        newFunc.getName(), callInputs);

    // Replace uses of outputs
    SmallVector<Value> outputs(data.outputs.begin(), data.outputs.end());
    for (unsigned i = 0; i < outputs.size(); ++i) {
      if (outputs[i]) {
        outputs[i].replaceAllUsesWith(callOp.getResult(i));
      }
    }

    // Safely erase operations in reverse order, avoiding double-free
    for (Operation *op : llvm::reverse(data.operations)) {
      if (op && erasedOperations.find(op) == erasedOperations.end()) {
        erasedOperations.insert(op);
        op->erase();
      }
    }

    // Safely erase movable operations that are no longer needed
    // BUT do not erase shared plaintext operations that are still used
    // elsewhere
    for (Operation *op : llvm::reverse(data.internalPlaintexts)) {
      if (op && erasedOperations.find(op) == erasedOperations.end()) {
        // Check if this plaintext operation has users outside the outlined
        // region
        bool hasExternalUsers = false;
        for (Value result : op->getResults()) {
          for (Operation *user : result.getUsers()) {
            // If any user is not in our region's operations, it's still needed
            // elsewhere
            bool userInRegion = false;
            for (Operation *regionOp : data.operations) {
              if (user == regionOp) {
                userInRegion = true;
                break;
              }
            }
            if (!userInRegion) {
              hasExternalUsers = true;
              break;
            }
          }
          if (hasExternalUsers) break;
        }

        // Only erase if no external users AND no remaining uses
        if (!hasExternalUsers && op->use_empty()) {
          erasedOperations.insert(op);
          op->erase();
          llvm::outs() << "DEBUG: Erased unused plaintext operation: "
                       << op->getName() << "\n";
        } else {
          llvm::outs() << "DEBUG: Keeping shared plaintext operation: "
                       << op->getName()
                       << " (has external users: " << hasExternalUsers
                       << ", remaining uses: " << !op->use_empty() << ")\n";
        }
      }
    }

    for (Operation *op : llvm::reverse(data.internalConstants)) {
      if (op && erasedOperations.find(op) == erasedOperations.end()) {
        // Check if this constant has users outside the outlined region
        bool hasExternalUsers = false;
        for (Value result : op->getResults()) {
          for (Operation *user : result.getUsers()) {
            // If any user is not in our region's operations, it's still needed
            // elsewhere
            bool userInRegion = false;
            for (Operation *regionOp : data.operations) {
              if (user == regionOp) {
                userInRegion = true;
                break;
              }
            }
            if (!userInRegion) {
              hasExternalUsers = true;
              break;
            }
          }
          if (hasExternalUsers) break;
        }

        // Only erase if no external users AND no remaining uses
        if (!hasExternalUsers && op->use_empty()) {
          erasedOperations.insert(op);
          op->erase();
          llvm::outs() << "DEBUG: Erased unused constant operation: "
                       << op->getName() << "\n";
        } else {
          llvm::outs() << "DEBUG: Keeping shared constant operation: "
                       << op->getName()
                       << " (has external users: " << hasExternalUsers
                       << ", remaining uses: " << !op->use_empty() << ")\n";
        }
      }
    }
  }

  void outlineRegions(std::vector<OutlineRegion> &regions,
                      func::FuncOp originalFunc, ModuleOp module) {
    std::vector<SafeRegionData> safeData;

    for (auto &region : regions) {
      if (!shouldOutlineRegion(region)) continue;

      safeData.emplace_back(region, originalFunc.getName());
      llvm::outs() << "Prepared region " << region.linearTransformId
                   << " for outlining (" << region.operations.size()
                   << " ops)\n";
    }

    std::vector<func::FuncOp> outlinedFuncs;
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

  std::vector<OutlineRegion> collectLinearTransformRegions(func::FuncOp func) {
    std::vector<OutlineRegion> regions;
    DenseMap<int64_t, OutlineRegion> regionMap;

    // Find start and end markers
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

    // Process complete regions
    for (auto &[regionId, region] : regionMap) {
      if (region.startOp && region.endOp) {
        collectOperationsBetween(region, func);
        // DON'T compute inputs/outputs yet - that happens after localization
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
};

}  // namespace heir
}  // namespace mlir
