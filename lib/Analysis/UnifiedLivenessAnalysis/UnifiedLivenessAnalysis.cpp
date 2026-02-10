#include "UnifiedLivenessAnalysis.h"

#include <algorithm>

#include "lib/Dialect/KMRT/IR/KMRTOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/IR/AsmState.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "unified-liveness-analysis"

namespace mlir {
namespace heir {

//===----------------------------------------------------------------------===//
// OperationLivenessInfo Implementation
//===----------------------------------------------------------------------===//

void OperationLivenessInfo::print(raw_ostream &os) const {
  os << "Timestep " << timestep << " (" << resultName << "): "
     << "Live CTs: " << liveCiphertextCount << ", Live Keys: " << liveKeyCount
     << ", Total Towers: " << totalTowers << "\n";
}

//===----------------------------------------------------------------------===//
// LivenessState Implementation
//===----------------------------------------------------------------------===//

unsigned LivenessState::getCiphertextCount() const {
  return ciphertextTowers.size();
}

unsigned LivenessState::getKeyCount() const { return liveKeys.size(); }

unsigned LivenessState::getCiphertextTowers() const {
  unsigned total = 0;
  for (const auto &[value, towers] : ciphertextTowers) {
    total += towers;
  }
  return total;
}

unsigned LivenessState::getKeyTowers() const {
  unsigned total = 0;
  for (const auto &[keyIndex, towers] : liveKeys) {
    total += towers;
  }
  return total;
}

unsigned LivenessState::getTotalTowers() const {
  return getCiphertextTowers() + getKeyTowers();
}

unsigned LivenessState::getMaxTowerNumber() const {
  unsigned maxTowers = 0;
  for (const auto &[value, towers] : ciphertextTowers) {
    maxTowers = std::max(maxTowers, towers);
  }
  for (const auto &[keyIndex, towers] : liveKeys) {
    maxTowers = std::max(maxTowers, towers);
  }
  return maxTowers;
}

// Helper function to extract MLIR result name from operation
static std::string getResultName(Operation *op) {
  std::string nameStr;
  llvm::raw_string_ostream nameStream(nameStr);

  // Try to get the name of the first result
  if (op->getNumResults() > 0) {
    Value firstResult = op->getResult(0);
    if (auto parentOp = firstResult.getParentRegion()->getParentOp()) {
      AsmState asmState(parentOp);
      firstResult.printAsOperand(nameStream, asmState);
      nameStream.flush();
    }
  } else {
    // For operations with no results, use operation name
    nameStream << op->getName();
    nameStream.flush();
  }

  return nameStr;
}
OperationLivenessInfo LivenessState::toOperationInfo(Operation *op) const {
  OperationLivenessInfo info(op);

  // Fill in operation identification - just the result name
  LLVM_DEBUG(info.resultName = getResultName(op););
  // timestep will be set by the caller

  // Fill in ciphertext information using clear function names
  info.liveCiphertextCount = getCiphertextCount();  // Number of ciphertexts
  info.totalCiphertextTowers =
      getCiphertextTowers();  // Total towers in ciphertexts
  for (const auto &[value, towers] : ciphertextTowers) {
    info.ciphertextTowerCounts.push_back(towers);
  }

  // Fill in key information using clear function names
  info.liveKeyCount = getKeyCount();     // Number of keys
  info.totalKeyTowers = getKeyTowers();  // Total towers in keys
  for (const auto &[keyIndex, towers] : liveKeys) {
    info.keyTowerCounts.push_back(towers);
    info.keyRotationIndices.push_back(keyIndex);
  }

  // Summary information
  info.maxTowerNumber = getMaxTowerNumber();
  info.totalTowers = getTotalTowers();

  return info;
}

void LivenessState::print(raw_ostream &os) const {
  os << "LivenessState(";
  os << "ciphertexts=" << getCiphertextCount() << " (" << getCiphertextTowers()
     << " towers)";
  os << ", keys=" << getKeyCount() << " (" << getKeyTowers() << " towers)";
  os << ", total=" << getTotalTowers() << " towers";
  os << ", max=" << getMaxTowerNumber() << " towers";
  os << ")";
}

//===----------------------------------------------------------------------===//
// UnifiedLivenessResults Implementation
//===----------------------------------------------------------------------===//

void UnifiedLivenessResults::addOperationInfo(
    const OperationLivenessInfo &info) {
  operationResults.push_back(info);
}

const OperationLivenessInfo *UnifiedLivenessResults::getInfoForOperation(
    Operation *op) const {
  for (const auto &info : operationResults) {
    if (info.op == op) {
      return &info;
    }
  }
  return nullptr;
}

unsigned UnifiedLivenessResults::getMaxTowerCount() const {
  unsigned maxTowers = 0;
  for (const auto &info : operationResults) {
    maxTowers = std::max(maxTowers, info.maxTowerNumber);
  }
  return maxTowers;
}

void UnifiedLivenessResults::print(raw_ostream &os) const {
  os << "UnifiedLivenessResults: " << operationResults.size()
     << " operations in execution order\n";
  for (const auto &info : operationResults) {
    os << "  Timestep " << info.timestep << " (" << info.resultName
       << "): " << info.liveCiphertextCount << " CTs ("
       << info.totalCiphertextTowers << " towers), " << info.liveKeyCount
       << " keys (" << info.totalKeyTowers << " towers)\n";
  }
}

//===----------------------------------------------------------------------===//
// Simple Timestep-Based Liveness Analysis
//===----------------------------------------------------------------------===//

// Helper to check if we should ignore this operation type
static bool shouldIgnoreOperation(Operation *op) {
  StringRef opName = op->getName().getStringRef();
  return opName.starts_with("arith.") || opName.starts_with("tensor.") ||
         opName.starts_with("func.") ||
         isa<openfhe::MakePackedPlaintextOp,
             openfhe::MakeCKKSPackedPlaintextOp>(op);
}

// Helper to check if this is a management operation (doesn't count as timestep)
static bool isManagementOperation(Operation *op) {
  return isa<kmrt::LoadKeyOp, kmrt::ClearKeyOp, openfhe::ClearCtOp>(op);
}

// Helper to check if this is a computational FHE operation (counts as timestep)
static bool isComputationalOperation(Operation *op) {
  return isa<openfhe::AddOp, openfhe::SubOp, openfhe::MulOp,
             openfhe::MulPlainOp, openfhe::RotOp, openfhe::BootstrapOp,
             openfhe::RelinOp, openfhe::ModReduceOp, openfhe::LevelReduceOp,
             openfhe::KeySwitchOp, openfhe::AutomorphOp, openfhe::ChebyshevOp>(
      op);
}

// Get towers from attributes or default
static unsigned getTowerCount(Operation *op, unsigned resultIndex = 0) {
  if (auto towersAttr = op->getAttrOfType<IntegerAttr>("result_towers")) {
    return towersAttr.getInt();
  }
  if (isa<openfhe::BootstrapOp>(op)) return 10;  // High level after bootstrap
  if (isa<openfhe::MulOp>(op)) return 5;         // Multiplication reduces level
  return 3;                                      // Default
}

// Extract key index from load key operation
static int64_t getKeyIndex(kmrt::LoadKeyOp loadOp) {
  if (auto indexAttr = loadOp->getAttrOfType<IntegerAttr>("index")) {
    return indexAttr.getInt();
  }
  // Try to get it from the operand directly
  if (auto indexOp = loadOp.getIndex().getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(indexOp.getValue())) {
      return intAttr.getInt();
    }
  }
  // Fallback: get the index attribute directly from the load operation
  return loadOp.getIndex()
      .getDefiningOp<arith::ConstantOp>()
      ->getAttrOfType<IntegerAttr>("value")
      .getInt();
}

// Extract key towers from load key operation
static unsigned getKeyTowers(kmrt::LoadKeyOp loadOp) {
  if (auto towersAttr = loadOp->getAttrOfType<IntegerAttr>("key_towers")) {
    return towersAttr.getInt();
  }
  return 32;  // Default key towers
}

UnifiedLivenessResults runUnifiedLivenessAnalysis(Operation *op) {
  UnifiedLivenessResults results;

  llvm::dbgs() << "=== Starting Simple Timestep-Based Liveness Analysis ===\n";

  // Current live state
  llvm::DenseSet<Value> liveCiphertexts;  // Currently live ciphertext values
  llvm::DenseMap<int64_t, unsigned> liveKeys;  // key_index -> tower_count

  unsigned timestep = 0;
  unsigned totalOpsProcessed = 0;

  op->walk([&](Operation *walkOp) {
    totalOpsProcessed++;

    // Skip operations we don't care about
    if (shouldIgnoreOperation(walkOp)) {
      return;
    }

    // Handle management operations (don't increment timestep)
    if (isManagementOperation(walkOp)) {
      if (auto loadOp = dyn_cast<kmrt::LoadKeyOp>(walkOp)) {
        // Add key to live set
        int64_t keyIndex = getKeyIndex(loadOp);
        unsigned towers = getKeyTowers(loadOp);
        liveKeys[keyIndex] = towers;

        llvm::dbgs() << "  Management: Added key " << keyIndex << " (" << towers
                     << " towers from key_towers attr)\n";
      } else if (auto clearOp = dyn_cast<kmrt::ClearKeyOp>(walkOp)) {
        // Remove key from live set
        Value keyToRemove = clearOp.getRotKey();
        if (auto loadOp = keyToRemove.getDefiningOp<kmrt::LoadKeyOp>()) {
          int64_t keyIndex = getKeyIndex(loadOp);
          liveKeys.erase(keyIndex);
          llvm::dbgs() << "  Management: Removed key " << keyIndex << "\n";
        }
      } else if (auto clearOp = dyn_cast<openfhe::ClearCtOp>(walkOp)) {
        // Remove ciphertext from live set when explicitly cleared
        Value ctToRemove = clearOp.getCiphertext();
        liveCiphertexts.erase(ctToRemove);
        LLVM_DEBUG(llvm::dbgs()
                       << "  Management: Cleared ciphertext "
                       << getResultName(ctToRemove.getDefiningOp()) << "\n";);
      }
      return;
    }

    // Only process computational operations as timesteps
    if (!isComputationalOperation(walkOp)) {
      return;
    }

    timestep++;

    // Key validation for rotation operations
    if (auto rotOp = dyn_cast<openfhe::RotOp>(walkOp)) {
      // Extract required key index from the rotation operation
      Value evalKey = rotOp.getEvalKey();
      if (auto loadOp = evalKey.getDefiningOp<kmrt::LoadKeyOp>()) {
        int64_t requiredKeyIndex = getKeyIndex(loadOp);

        if (liveKeys.find(requiredKeyIndex) == liveKeys.end()) {
          llvm::dbgs() << "ERROR: Rotation operation " << timestep
                       << " requires key " << requiredKeyIndex
                       << " but it is not live!\n";
          llvm::dbgs() << "Available keys: [";
          for (auto &[keyIdx, towers] : liveKeys) {
            llvm::dbgs() << keyIdx << " ";
          }
          llvm::dbgs() << "]\n";

          walkOp->emitError("Rotation operation requires key index ")
              << requiredKeyIndex << " but it is not live in crypto context";
          return;
        }
      }
    }

    // Ciphertext removal is handled by explicit clear_ct operations
    // inserted by the InsertCiphertextClears pass - no naive removal here

    // Add new ciphertext results
    for (Value result : walkOp->getResults()) {
      if (isa<lwe::NewLWECiphertextType, lwe::LWECiphertextType>(
              result.getType())) {
        liveCiphertexts.insert(result);
      }
    }

    // Create current state for recording
    LivenessState currentState;

    // Add live ciphertexts to state
    for (Value ct : liveCiphertexts) {
      unsigned towers = getTowerCount(ct.getDefiningOp());
      currentState.ciphertextTowers[ct] = towers;
    }

    // Add all live keys to state
    currentState.liveKeys = liveKeys;

    // Record this timestep
    OperationLivenessInfo info = currentState.toOperationInfo(walkOp);
    info.timestep = timestep;  // Set the timestep number
    results.addOperationInfo(info);

    // Enhanced debug output with result name - showing counts AND towers
    // clearly
    if (timestep <= 20 || timestep % 100 == 0) {
      LLVM_DEBUG(std::string resultName = getResultName(walkOp););

      // Calculate total towers for debug using clear function names
      unsigned totalCipherTowers = 0;
      for (Value ct : liveCiphertexts) {
        totalCipherTowers += getTowerCount(ct.getDefiningOp());
      }
      unsigned totalKeyTowers = 0;
      for (auto &[keyIdx, towers] : liveKeys) {
        totalKeyTowers += towers;
      }

      llvm::dbgs() << "Timestep " << timestep << " (" << walkOp->getName()
                   << " -> ";
      // LLVM_DEBUG(<< resultName);
      llvm::dbgs() << "): " << liveCiphertexts.size() << " CTs ("
                   << totalCipherTowers << " towers), " << liveKeys.size()
                   << " keys (" << totalKeyTowers << " towers)";
      if (isa<openfhe::BootstrapOp>(walkOp)) {
        llvm::dbgs() << " [BOOTSTRAP]";
      }
      llvm::dbgs() << "\n";
    }
  });

  llvm::dbgs() << "=== Analysis Complete: " << timestep << " timesteps, "
               << totalOpsProcessed << " total operations processed ===\n";
  return results;
}

}  // namespace heir
}  // namespace mlir
