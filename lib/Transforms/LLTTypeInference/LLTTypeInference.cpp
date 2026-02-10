#include "lib/Transforms/LLTTypeInference/LLTTypeInference.h"

#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "lib/Dialect/CKKS/IR/CKKSOps.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project

#define DEBUG_TYPE "llt-type-inf"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_LLTTYPEINFERENCE
#include "lib/Transforms/LLTTypeInference/LLTTypeInference.h.inc"

static int64_t getScalingFactor(Type type) {
  if (auto ctType = dyn_cast<lwe::NewLWECiphertextType>(type)) {
    auto ptSpace = ctType.getPlaintextSpace();
    if (auto encoding = dyn_cast<lwe::InverseCanonicalEncodingAttr>(
            ptSpace.getEncoding())) {
      return encoding.getScalingFactor();
    }
  } else if (auto ptType = dyn_cast<lwe::NewLWEPlaintextType>(type)) {
    auto ptSpace = ptType.getPlaintextSpace();
    if (auto encoding = dyn_cast<lwe::InverseCanonicalEncodingAttr>(
            ptSpace.getEncoding())) {
      return encoding.getScalingFactor();
    }
  }
  return 0;
}

static Type createTypeWithScalingFactor(Type originalType, int64_t newScale,
                                        MLIRContext *context) {
  if (auto ctType = dyn_cast<lwe::NewLWECiphertextType>(originalType)) {
    auto newEncoding = lwe::InverseCanonicalEncodingAttr::get(context, newScale);
    auto newPtSpace = lwe::PlaintextSpaceAttr::get(
        context, ctType.getPlaintextSpace().getRing(), newEncoding);

    return lwe::NewLWECiphertextType::get(
        context, ctType.getApplicationData(), newPtSpace,
        ctType.getCiphertextSpace(), ctType.getKey(), ctType.getModulusChain());
  }
  return originalType;
}

struct LLTTypeInference : impl::LLTTypeInferenceBase<LLTTypeInference> {
  using LLTTypeInferenceBase::LLTTypeInferenceBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto context = moduleOp->getContext();
    
    llvm::dbgs() << "=== MODULE-LEVEL TYPE INFERENCE ===\n";
    
    // Walk all functions in the module
    moduleOp->walk([&](func::FuncOp funcOp) {
      llvm::dbgs() << "Processing function: " << funcOp.getName() << "\n";
      
      // Build a map of what each value's scale should be after simulation
      DenseMap<Value, int64_t> simulatedScales;
      
      // Initialize function arguments
      for (auto arg : funcOp.getArguments()) {
        simulatedScales[arg] = getScalingFactor(arg.getType());
        llvm::dbgs() << "Function arg scale: " << simulatedScales[arg] << "\n";
      }
      
      // Process operations in order and simulate their scaling behavior
      funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (op == funcOp.getOperation()) return;
        
        if (auto linearOp = dyn_cast<ckks::LinearTransformOp>(op)) {
          // Simulate the lowering
          Value input = linearOp.getInput();
          int64_t inputScale = simulatedScales.lookup(input);
          int64_t weightsScale = getScalingFactor(linearOp.getWeights().getType());
          
          // Simulate: ckks.mul_plain operations produce inputScale + weightsScale
          // Simulate: ckks.add operations preserve the scale (all inputs same)
          int64_t simulatedResultScale = inputScale + weightsScale;
          
          llvm::dbgs() << "Simulating linear_transform: input=" << inputScale
                       << " + weights=" << weightsScale 
                       << " = " << simulatedResultScale << "\n";
          
          // Store the simulated scale
          simulatedScales[linearOp.getResult()] = simulatedResultScale;
          
          // Update the actual result type
          int64_t currentScale = getScalingFactor(linearOp.getResult().getType());
          if (currentScale != simulatedResultScale) {
            llvm::dbgs() << "  Updating result type: " << currentScale 
                         << " -> " << simulatedResultScale << "\n";
            Type newResultType = createTypeWithScalingFactor(
                linearOp.getResult().getType(), simulatedResultScale, context);
            linearOp.getResult().setType(cast<lwe::NewLWECiphertextType>(newResultType));
          }
        }
        
        else if (auto addOp = dyn_cast<ckks::AddOp>(op)) {
          // Add operations: result scale = max(input scales)
          int64_t lhsScale = simulatedScales.lookup(addOp.getLhs());
          int64_t rhsScale = simulatedScales.lookup(addOp.getRhs());
          int64_t resultScale = std::max(lhsScale, rhsScale);
          
          simulatedScales[addOp.getResult()] = resultScale;
          
          // Update the actual result type
          int64_t currentScale = getScalingFactor(addOp.getResult().getType());
          if (currentScale != resultScale) {
            llvm::dbgs() << "  Updating add result: " << currentScale 
                         << " -> " << resultScale << "\n";
            Type newResultType = createTypeWithScalingFactor(
                addOp.getResult().getType(), resultScale, context);
            addOp.getResult().setType(cast<lwe::NewLWECiphertextType>(newResultType));
          }
        }
        
        else if (auto mulOp = dyn_cast<ckks::MulOp>(op)) {
          // Multiply operations: result scale = lhs scale + rhs scale
          int64_t lhsScale = simulatedScales.lookup(mulOp.getLhs());
          int64_t rhsScale = simulatedScales.lookup(mulOp.getRhs());
          int64_t resultScale = lhsScale + rhsScale;
          
          simulatedScales[mulOp.getResult()] = resultScale;
          
          int64_t currentScale = getScalingFactor(mulOp.getResult().getType());
          if (currentScale != resultScale) {
            llvm::dbgs() << "  Updating mul result: " << currentScale 
                         << " -> " << resultScale << "\n";
            Type newResultType = createTypeWithScalingFactor(
                mulOp.getResult().getType(), resultScale, context);
            mulOp.getResult().setType(cast<lwe::NewLWECiphertextType>(newResultType));
          }
        }
        
        else if (auto rotateOp = dyn_cast<ckks::RotateOp>(op)) {
          // Rotate preserves scale
          int64_t inputScale = simulatedScales.lookup(rotateOp.getInput());
          simulatedScales[rotateOp.getResult()] = inputScale;
          
          Type inputType = rotateOp.getInput().getType();
          if (rotateOp.getResult().getType() != inputType) {
            llvm::dbgs() << "  Updating rotate result type\n";
            rotateOp.getResult().setType(cast<lwe::NewLWECiphertextType>(inputType));
          }
        }
        
        else if (auto addPlainOp = dyn_cast<ckks::AddPlainOp>(op)) {
          // Add plain preserves ciphertext scale, but plaintext must match
          Value ctInput, ptInput;
          if (isa<lwe::NewLWECiphertextType>(addPlainOp.getLhs().getType())) {
            ctInput = addPlainOp.getLhs();
            ptInput = addPlainOp.getRhs();
          } else {
            ctInput = addPlainOp.getRhs();
            ptInput = addPlainOp.getLhs();
          }
          
          // Get ciphertext scale (use actual type if not in simulation map)
          int64_t ctScale = simulatedScales.lookup(ctInput);
          if (ctScale == 0) {
            ctScale = getScalingFactor(ctInput.getType());
            simulatedScales[ctInput] = ctScale;
            llvm::dbgs() << "  Missing ciphertext scale, using actual: " << ctScale << "\n";
          }
          
          int64_t ptScale = getScalingFactor(ptInput.getType());
          
          llvm::dbgs() << "Processing add_plain: ct_scale=" << ctScale 
                       << ", pt_scale=" << ptScale << "\n";
          
          // Update plaintext to match ciphertext scale
          if (ptScale != ctScale) {
            llvm::dbgs() << "  Updating plaintext scale for add_plain: " 
                         << ptScale << " -> " << ctScale << "\n";
            Type newPtType = createTypeWithScalingFactor(ptInput.getType(), ctScale, context);
            ptInput.setType(cast<lwe::NewLWEPlaintextType>(newPtType));
          }
          
          simulatedScales[addPlainOp.getResult()] = ctScale;
          
          int64_t currentScale = getScalingFactor(addPlainOp.getResult().getType());
          if (currentScale != ctScale) {
            llvm::dbgs() << "  Updating add_plain result: " << currentScale 
                         << " -> " << ctScale << "\n";
            Type newResultType = createTypeWithScalingFactor(
                addPlainOp.getResult().getType(), ctScale, context);
            addPlainOp.getResult().setType(cast<lwe::NewLWECiphertextType>(newResultType));
          }
        }
        
        else if (auto mulPlainOp = dyn_cast<ckks::MulPlainOp>(op)) {
          // Mul plain: result scale = ct scale + pt scale
          Value ctInput, ptInput;
          if (isa<lwe::NewLWECiphertextType>(mulPlainOp.getLhs().getType())) {
            ctInput = mulPlainOp.getLhs();
            ptInput = mulPlainOp.getRhs();
          } else {
            ctInput = mulPlainOp.getRhs();
            ptInput = mulPlainOp.getLhs();
          }
          
          int64_t ctScale = simulatedScales.lookup(ctInput);
          int64_t ptScale = getScalingFactor(ptInput.getType());
          int64_t resultScale = ctScale + ptScale;
          
          simulatedScales[mulPlainOp.getResult()] = resultScale;
          
          int64_t currentScale = getScalingFactor(mulPlainOp.getResult().getType());
          if (currentScale != resultScale) {
            llvm::dbgs() << "  Updating mul_plain result: " << currentScale 
                         << " -> " << resultScale << "\n";
            Type newResultType = createTypeWithScalingFactor(
                mulPlainOp.getResult().getType(), resultScale, context);
            mulPlainOp.getResult().setType(cast<lwe::NewLWECiphertextType>(newResultType));
          }
        }
        
        // Handle bootstrap operations (reset scale)
        else if (auto bootstrapOp = dyn_cast<ckks::BootstrapOp>(op)) {
          // Bootstrap typically resets to the initial scale
          int64_t inputScale = simulatedScales.lookup(bootstrapOp.getInput());
          simulatedScales[bootstrapOp.getResult()] = inputScale;
          
          Type inputType = bootstrapOp.getInput().getType();
          if (bootstrapOp.getResult().getType() != inputType) {
            llvm::dbgs() << "  Updating bootstrap result type\n";
            bootstrapOp.getResult().setType(cast<lwe::NewLWECiphertextType>(inputType));
          }
        }
        
        // Handle other operations that might not have simulated scales yet
        for (Value result : op->getResults()) {
          if (simulatedScales.find(result) == simulatedScales.end()) {
            // Default: use the current scale from the type
            simulatedScales[result] = getScalingFactor(result.getType());
          }
        }
      });
      
      llvm::dbgs() << "Completed processing function: " << funcOp.getName() << "\n";
    });
    
    llvm::dbgs() << "=== SIMULATION-BASED TYPE INFERENCE COMPLETED ===\n";
  }
};

}  // namespace heir
}  // namespace mlir
