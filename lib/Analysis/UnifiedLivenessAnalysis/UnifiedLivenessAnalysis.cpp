#include "UnifiedLivenessAnalysis.h"

#include <algorithm>

#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"           // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

#define DEBUG_TYPE "unified-liveness-analysis"

namespace mlir {
namespace heir {

//===----------------------------------------------------------------------===//
// LivenessState Implementation - Simple like RotationDepthAnalysis
//===----------------------------------------------------------------------===//

unsigned LivenessState::getTotalTowers() const {
  unsigned total = 0;
  for (const auto &[value, towers] : ciphertextTowers) {
    total += towers;
  }
  for (const auto &[value, towers] : keyTowers) {
    total += towers;
  }
  return total;
}

unsigned LivenessState::getMaxTowerNumber() const {
  unsigned maxTowers = 0;
  for (const auto &[value, towers] : ciphertextTowers) {
    maxTowers = std::max(maxTowers, towers);
  }
  for (const auto &[value, towers] : keyTowers) {
    maxTowers = std::max(maxTowers, towers);
  }
  return maxTowers;
}

unsigned LivenessState::getCiphertextCount() const {
  return ciphertextTowers.size();
}

unsigned LivenessState::getKeyCount() const { return keyTowers.size(); }

bool LivenessState::operator==(const LivenessState &other) const {
  // Simple comparison like in existing analyses
  return ciphertextTowers.size() == other.ciphertextTowers.size() &&
         keyTowers.size() == other.keyTowers.size();
}

LivenessState LivenessState::join(const LivenessState &lhs,
                                  const LivenessState &rhs) {
  LivenessState result = lhs;

  // Merge ciphertext towers - keep higher values (conservative)
  for (const auto &[value, towers] : rhs.ciphertextTowers) {
    auto it = result.ciphertextTowers.find(value);
    if (it == result.ciphertextTowers.end()) {
      result.ciphertextTowers[value] = towers;
    } else {
      result.ciphertextTowers[value] = std::max(it->second, towers);
    }
  }

  // Merge key towers
  for (const auto &[value, towers] : rhs.keyTowers) {
    auto it = result.keyTowers.find(value);
    if (it == result.keyTowers.end()) {
      result.keyTowers[value] = towers;
    } else {
      result.keyTowers[value] = std::max(it->second, towers);
    }
  }

  // Merge key flags
  for (const auto &[value, isKey] : rhs.isKeyMap) {
    result.isKeyMap[value] = isKey;
  }

  return result;
}

void LivenessState::print(raw_ostream &os) const {
  os << "LivenessState(";
  os << "total_towers=" << getTotalTowers();
  os << ", max_towers=" << getMaxTowerNumber();
  os << ", ciphertexts=" << getCiphertextCount();
  os << ", keys=" << getKeyCount();
  os << ")";
}

//===----------------------------------------------------------------------===//
// UnifiedLivenessAnalysis Implementation
//===----------------------------------------------------------------------===//

LogicalResult UnifiedLivenessAnalysis::visitOperation(
    Operation *op, ArrayRef<const LivenessLattice *> operands,
    ArrayRef<LivenessLattice *> results) {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing op: " << *op << "\n");

  // Start with the current state from operands
  LivenessState currentState;
  if (!operands.empty() && operands[0]) {
    currentState = operands[0]->getValue();
  }

  // Utility to propagate state to all results
  auto propagateToResults = [&](const LivenessState &state) {
    for (auto *result : results) {
      ChangeResult changed = result->join(state);
      propagateIfChanged(result, changed);
    }
  };

  return llvm::TypeSwitch<Operation &, LogicalResult>(*op)
      // Handle key deserialize operations
      .Case<openfhe::DeserializeKeyOp>([&](auto deserOp) {
        Value evalKey = deserOp.getResult();
        unsigned towers = getKeyTowerCount(evalKey);
        int64_t rotationIndex = getKeyRotationIndex(deserOp);
        unsigned keyDepth = getKeyDepthFromDeserialize(deserOp);

        currentState.keyTowers[evalKey] = towers;
        currentState.isKeyMap[evalKey] = true;
        currentState.keyRotationIndices[evalKey] = rotationIndex;
        currentState.keyDepths[evalKey] = keyDepth;

        LLVM_DEBUG(llvm::dbgs()
                   << "  Added live key with " << towers << " towers\n");
        propagateToResults(currentState);
        return success();
      })

      // Handle key clearing operations
      .Case<openfhe::ClearKeyOp>([&](auto clearOp) {
        Value evalKey = clearOp.getEvalKey();
        currentState.keyTowers.erase(evalKey);
        currentState.isKeyMap.erase(evalKey);
        currentState.keyRotationIndices.erase(evalKey);
        currentState.keyDepths.erase(evalKey);

        LLVM_DEBUG(llvm::dbgs() << "  Cleared key from liveness\n");
        propagateToResults(currentState);
        return success();
      })

      // Handle ciphertext-producing operations
      .Case<openfhe::AddOp, openfhe::MulOp, openfhe::RotOp,
            openfhe::BootstrapOp, openfhe::RelinOp, openfhe::ModReduceOp>(
          [&](Operation *cipherOp) {
            // Add the result ciphertext to live set
            if (cipherOp->getNumResults() > 0) {
              Value result = cipherOp->getResult(0);
              unsigned towers = getCiphertextTowerCount(result);

              currentState.ciphertextTowers[result] = towers;
              currentState.isKeyMap[result] = false;

              LLVM_DEBUG(llvm::dbgs() << "  Added live ciphertext with "
                                      << towers << " towers\n");
            }

            // Remove consumed operands if they have only one use
            for (Value operand : cipherOp->getOperands()) {
              if (operand.hasOneUse()) {
                currentState.ciphertextTowers.erase(operand);
                currentState.keyTowers.erase(operand);
                currentState.isKeyMap.erase(operand);
                currentState.keyRotationIndices.erase(operand);
                currentState.keyDepths.erase(operand);
              }
            }

            propagateToResults(currentState);
            return success();
          })

      // Handle operations that may end ciphertext lifetime
      .Case<func::ReturnOp>([&](auto returnOp) {
        // Remove returned values from live set as they leave the function
        for (Value operand : returnOp.getOperands()) {
          currentState.ciphertextTowers.erase(operand);
          currentState.keyTowers.erase(operand);
          currentState.isKeyMap.erase(operand);
          currentState.keyRotationIndices.erase(operand);
          currentState.keyDepths.erase(operand);
        }

        propagateToResults(currentState);
        return success();
      })

      // Handle general operations
      .Default([&](Operation &defaultOp) {
        // Remove operands with single use
        for (Value operand : defaultOp.getOperands()) {
          if (operand.hasOneUse()) {
            currentState.ciphertextTowers.erase(operand);
            currentState.keyTowers.erase(operand);
            currentState.isKeyMap.erase(operand);
            currentState.keyRotationIndices.erase(operand);
            currentState.keyDepths.erase(operand);
          }
        }

        // Add any new results to the live set
        for (Value result : defaultOp.getResults()) {
          if (isCiphertextType(result.getType())) {
            unsigned towers = getCiphertextTowerCount(result);
            currentState.ciphertextTowers[result] = towers;
            currentState.isKeyMap[result] = false;
          } else if (isKeyType(result.getType())) {
            unsigned towers = getKeyTowerCount(result);
            currentState.keyTowers[result] = towers;
            currentState.isKeyMap[result] = true;
          }
        }

        propagateToResults(currentState);
        return success();
      });
}

void UnifiedLivenessAnalysis::setToEntryState(LivenessLattice *lattice) {
  propagateIfChanged(lattice, lattice->join(LivenessState()));
}

bool UnifiedLivenessAnalysis::isCiphertextType(Type type) const {
  return isa<lwe::NewLWECiphertextType, lwe::LWECiphertextType>(type);
}

bool UnifiedLivenessAnalysis::isKeyType(Type type) const {
  return isa<openfhe::EvalKeyType>(type);
}

unsigned UnifiedLivenessAnalysis::getCiphertextTowerCount(Value value) const {
  // Get tower count from result_towers attribute if available
  if (auto defOp = value.getDefiningOp()) {
    if (auto towersAttr = defOp->getAttrOfType<IntegerAttr>("result_towers")) {
      return towersAttr.getInt();
    }
  }

  // Fallback: try to infer from operation type
  if (auto defOp = value.getDefiningOp()) {
    // Multiplication typically increases tower count
    if (isa<openfhe::MulOp>(defOp)) {
      return 5;  // Conservative estimate
    }
    // Bootstrap resets to high level
    if (isa<openfhe::BootstrapOp>(defOp)) {
      return 10;  // Conservative estimate
    }
  }

  return 3;  // Default conservative estimate
}

unsigned UnifiedLivenessAnalysis::getKeyTowerCount(Value value,
                                                   unsigned multDepth) const {
  if (auto defOp = value.getDefiningOp()) {
    if (auto deserializeOp = dyn_cast<openfhe::DeserializeKeyOp>(defOp)) {
      unsigned depth = getKeyDepthFromDeserialize(deserializeOp);
      unsigned qTowers = calculateKeyQTowers(multDepth, depth);
      return qTowers + FIXED_P_TOWERS;
    }
  }

  // Default fallback
  return FIXED_P_TOWERS + 3;
}

unsigned UnifiedLivenessAnalysis::getKeyDepthFromDeserialize(
    Operation *deserializeOp) const {
  if (auto depthAttr = deserializeOp->getAttrOfType<IntegerAttr>("key_depth")) {
    return depthAttr.getInt();
  }
  return 0;  // Default depth
}

int64_t UnifiedLivenessAnalysis::getKeyRotationIndex(
    Operation *deserializeOp) const {
  if (auto deserOp = dyn_cast<openfhe::DeserializeKeyOp>(deserializeOp)) {
    return deserOp.getEvalKey().getType().getRotationIndex().getInt();
  }
  return 0;  // Default rotation index
}

unsigned UnifiedLivenessAnalysis::calculateKeyQTowers(unsigned multDepth,
                                                      unsigned level) const {
  // Formula: mult_depth - level + 2
  if (level > multDepth + 2) {
    return 1;  // Minimum towers
  }
  return multDepth - level + 2;
}

//===----------------------------------------------------------------------===//
// Helper Function Implementation
//===----------------------------------------------------------------------===//

UnifiedLivenessResults runUnifiedLivenessAnalysis(Operation *op) {
  UnifiedLivenessResults results;

  // Set up dataflow solver
  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
  solver.load<UnifiedLivenessAnalysis>();

  if (failed(solver.initializeAndRun(op))) {
    // Return empty results on failure
    return results;
  }

  // Walk operations and collect results
  op->walk([&](Operation *walkOp) {
    // For now, create a mock state since we can't easily query the solver
    // This will be replaced with proper solver querying later
    LivenessState mockState;

    // Add heuristic data based on operation type
    if (isa<openfhe::AddOp, openfhe::MulOp, openfhe::RotOp>(walkOp)) {
      // Mock some live ciphertexts
      for (Value operand : walkOp->getOperands()) {
        if (isa<lwe::NewLWECiphertextType, lwe::LWECiphertextType>(
                operand.getType())) {
          mockState.ciphertextTowers[operand] = 3;  // Conservative estimate
          mockState.isKeyMap[operand] = false;
        }
      }
    }

    if (auto deserOp = dyn_cast<openfhe::DeserializeKeyOp>(walkOp)) {
      Value evalKey = deserOp.getResult();
      mockState.keyTowers[evalKey] = 7;  // Conservative estimate
      mockState.isKeyMap[evalKey] = true;
      mockState.keyRotationIndices[evalKey] =
          deserOp.getEvalKey().getType().getRotationIndex().getInt();
      mockState.keyDepths[evalKey] = 0;  // Default for now
    }

    // Convert to structured result
    if (!mockState.ciphertextTowers.empty() || !mockState.keyTowers.empty()) {
      OperationLivenessInfo info = mockState.toOperationInfo(walkOp);
      results.addOperationInfo(info);
    }
  });

  return results;
}

}  // namespace heir
}  // namespace mlir
