#include "UnifiedLivenessAnalysis.h"

#include <algorithm>

#include "lib/Dialect/KMRT/IR/KMRTOps.h"
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
// LivenessState Implementation
//===----------------------------------------------------------------------===//

unsigned LivenessState::getTotalTowers() const {
  unsigned total = 0;
  for (const auto &[value, towers] : ciphertextTowers) {
    total += towers;
  }
  for (const auto &[value, towers] : keyTowers) {
    total += towers;
  }
  for (const auto &[rotIndex, towers] : globalKeyTowers) {
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
  for (const auto &[rotIndex, towers] : globalKeyTowers) {
    maxTowers = std::max(maxTowers, towers);
  }
  return maxTowers;
}

unsigned LivenessState::getCiphertextCount() const {
  return ciphertextTowers.size();
}

unsigned LivenessState::getKeyCount() const {
  return keyTowers.size() + globalKeyTowers.size();
}

bool LivenessState::operator==(const LivenessState &other) const {
  return ciphertextTowers == other.ciphertextTowers &&
         keyTowers == other.keyTowers && isKeyMap == other.isKeyMap &&
         keyRotationIndices == other.keyRotationIndices &&
         keyDepths == other.keyDepths &&
         globalKeyTowers == other.globalKeyTowers;
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

  // Merge other key information
  for (const auto &[value, isKey] : rhs.isKeyMap) {
    result.isKeyMap[value] = isKey;
  }

  for (const auto &[value, rotIndex] : rhs.keyRotationIndices) {
    result.keyRotationIndices[value] = rotIndex;
  }

  for (const auto &[value, depth] : rhs.keyDepths) {
    result.keyDepths[value] = depth;
  }

  for (const auto &[rotIndex, towers] : rhs.globalKeyTowers) {
    result.globalKeyTowers[rotIndex] =
        std::max(result.globalKeyTowers[rotIndex], towers);
  }

  return result;
}

OperationLivenessInfo LivenessState::toOperationInfo(Operation *op) const {
  OperationLivenessInfo info(op);

  // Fill in ciphertext information
  info.liveCiphertextCount = getCiphertextCount();
  for (const auto &[value, towers] : ciphertextTowers) {
    info.ciphertextTowerCounts.push_back(towers);
    info.totalCiphertextTowers += towers;
  }

  // Fill in key information from both SSA keys and global keys
  info.liveKeyCount = getKeyCount();

  // SSA keys (explicit evaluation keys)
  for (const auto &[value, towers] : keyTowers) {
    info.keyTowerCounts.push_back(towers);
    info.totalKeyTowers += towers;

    // Add rotation index if available
    auto rotIt = keyRotationIndices.find(value);
    if (rotIt != keyRotationIndices.end()) {
      info.keyRotationIndices.push_back(rotIt->second);
    }

    // Add depth if available
    auto depthIt = keyDepths.find(value);
    if (depthIt != keyDepths.end()) {
      info.keyDepths.push_back(depthIt->second);
    }
  }

  // Global keys (bootstrap keys stored in crypto context)
  for (const auto &[rotIndex, towers] : globalKeyTowers) {
    info.keyTowerCounts.push_back(towers);
    info.totalKeyTowers += towers;
    info.keyRotationIndices.push_back(rotIndex);
    info.keyDepths.push_back(0);  // Default depth for global keys
  }

  // Summary information
  info.maxTowerNumber = getMaxTowerNumber();
  info.totalTowers = getTotalTowers();

  return info;
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
     << " operations\n";
  for (const auto &info : operationResults) {
    os << "  Op: " << info.op->getName()
       << ", Live CTs: " << info.liveCiphertextCount
       << ", Live Keys: " << info.liveKeyCount
       << ", Total Towers: " << info.totalTowers << "\n";
  }
}

//===----------------------------------------------------------------------===//
// Unused dataflow analysis methods - kept for interface compatibility
//===----------------------------------------------------------------------===//

LogicalResult UnifiedLivenessAnalysis::visitOperation(
    Operation *op, ArrayRef<const LivenessLattice *> operands,
    ArrayRef<LivenessLattice *> results) {
  return success();
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
  return 3;  // Unused
}

unsigned UnifiedLivenessAnalysis::getKeyTowerCount(Value value,
                                                   unsigned multDepth) const {
  return 7;  // Unused
}

unsigned UnifiedLivenessAnalysis::getKeyDepthFromDeserialize(
    Operation *deserializeOp) const {
  return 0;  // Unused
}

int64_t UnifiedLivenessAnalysis::getKeyRotationIndex(
    Operation *deserializeOp) const {
  return 0;  // Unused
}

unsigned UnifiedLivenessAnalysis::calculateKeyQTowers(unsigned multDepth,
                                                      unsigned level) const {
  return 3;  // Unused
}

//===----------------------------------------------------------------------===//
// Simple Timestep-Based Liveness Analysis
//===----------------------------------------------------------------------===//

// Helper to check if we should ignore this operation type
static bool shouldIgnoreOperation(Operation *op) {
  // Ignore arith, tensor, and other non-FHE operations
  StringRef opName = op->getName().getStringRef();
  return opName.starts_with("arith.") || opName.starts_with("tensor.") ||
         opName.starts_with("func.") ||
         isa<openfhe::MakePackedPlaintextOp,
             openfhe::MakeCKKSPackedPlaintextOp>(op);
}

// Helper to check if this is a management operation (doesn't count as timestep)
static bool isManagementOperation(Operation *op) {
  return isa<openfhe::DeserializeKeyOp, openfhe::ClearKeyOp,
             openfhe::ClearCtOp>(op);
}

// Helper to check if this is a computational FHE operation (counts as timestep)
static bool isComputationalOperation(Operation *op) {
  return isa<openfhe::AddOp, openfhe::SubOp, openfhe::MulOp,
             openfhe::MulPlainOp, openfhe::RotOp, openfhe::BootstrapOp,
             openfhe::RelinOp, openfhe::ModReduceOp, openfhe::LevelReduceOp,
             openfhe::KeySwitchOp, openfhe::AutomorphOp, openfhe::ChebyshevOp>(
      op);
}

// Get towers from result_towers attribute or default
static unsigned getTowerCount(Operation *op, unsigned resultIndex = 0) {
  if (auto towersAttr = op->getAttrOfType<IntegerAttr>("result_towers")) {
    return towersAttr.getInt();
  }

  // Default based on operation type
  if (isa<openfhe::BootstrapOp>(op)) return 10;  // High level after bootstrap
  if (isa<openfhe::MulOp>(op)) return 5;         // Multiplication reduces level
  return 3;                                      // Default
}

UnifiedLivenessResults runUnifiedLivenessAnalysis(Operation *op) {
  UnifiedLivenessResults results;

  llvm::errs() << "=== Starting Simple Timestep-Based Liveness Analysis ===\n";

  // Current live state
  llvm::DenseSet<Value> liveCiphertexts;  // Currently live ciphertext values
  llvm::DenseMap<int64_t, unsigned> liveKeys;  // rotation_index -> tower_count

  unsigned timestep = 0;
  unsigned totalOpsProcessed = 0;

  op->walk([&](Operation *walkOp) {
    totalOpsProcessed++;

    // Skip operations we don't care about
    if (shouldIgnoreOperation(walkOp)) {
      return;
    }

    // Handle management operations (update state but don't count as timestep)
    if (isManagementOperation(walkOp)) {
      llvm::TypeSwitch<Operation *>(walkOp)
          .Case<openfhe::DeserializeKeyOp>(
              [&](openfhe::DeserializeKeyOp deserOp) {
                if (auto indexAttr =
                        deserOp->getAttrOfType<IntegerAttr>("index")) {
                  int64_t keyIndex = indexAttr.getInt();
                  unsigned towers = 7;  // Default fallback

                  // First priority: use exact key_towers attribute if available
                  if (auto keyTowersAttr =
                          deserOp->getAttrOfType<IntegerAttr>("key_towers")) {
                    towers = keyTowersAttr.getInt();
                    llvm::errs()
                        << "  Management: Added key " << keyIndex << " ("
                        << towers << " towers from key_towers attr)\n";
                  }
                  // Second priority: calculate from key_depth attribute
                  else if (auto depthAttr = deserOp->getAttrOfType<IntegerAttr>(
                               "key_depth")) {
                    unsigned depth = depthAttr.getInt();
                    unsigned qTowers = std::max(1u, 10u - depth + 2u);
                    towers = qTowers + 4;  // Q towers + 4 P towers
                    llvm::errs()
                        << "  Management: Added key " << keyIndex << " ("
                        << towers << " towers from depth " << depth << ")\n";
                  }
                  // Fallback: default estimate
                  else {
                    llvm::errs() << "  Management: Added key " << keyIndex
                                 << " (" << towers << " towers - default)\n";
                  }

                  liveKeys[keyIndex] = towers;
                }
              })
          .Case<openfhe::ClearKeyOp>([&](openfhe::ClearKeyOp clearOp) {
            if (auto evalKey = clearOp.getEvalKey()) {
              if (auto deserOp =
                      evalKey.getDefiningOp<openfhe::DeserializeKeyOp>()) {
                if (auto indexAttr =
                        deserOp->getAttrOfType<IntegerAttr>("index")) {
                  int64_t keyIndex = indexAttr.getInt();
                  liveKeys.erase(keyIndex);
                  llvm::errs()
                      << "  Management: Removed key " << keyIndex << "\n";
                }
              }
            }
          })
          .Case<openfhe::ClearCtOp>([&](openfhe::ClearCtOp clearOp) {
            Value ct = clearOp.getCiphertext();
            liveCiphertexts.erase(ct);
            llvm::errs() << "  Management: Cleared ciphertext\n";
          });
      return;  // Don't count as timestep
    }

    // Handle computational operations (count as timestep)
    if (isComputationalOperation(walkOp)) {
      timestep++;

      // CRITICAL VALIDATION: Check rotation operations have required keys
      if (auto rotOp = dyn_cast<openfhe::RotOp>(walkOp)) {
        // Get the rotation index from the evaluation key operand
        Value evalKey = rotOp.getEvalKey();
        if (auto deserOp = evalKey.getDefiningOp<openfhe::DeserializeKeyOp>()) {
          if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
            int64_t requiredKeyIndex = indexAttr.getInt();

            if (liveKeys.find(requiredKeyIndex) == liveKeys.end()) {
              llvm::errs() << "FATAL ERROR: Rotation operation at timestep "
                           << timestep << " requires key " << requiredKeyIndex
                           << " but it is not live! Available keys: [";
              for (auto &[keyIdx, towers] : liveKeys) {
                llvm::errs() << keyIdx << " ";
              }
              llvm::errs() << "]\n";

              // This is a hard error - the program cannot execute
              walkOp->emitError("Rotation operation requires key index ")
                  << requiredKeyIndex
                  << " but it is not live in crypto context";
              return;
            }
          }
        }
      }

      // Debug what operations we're seeing after timestep 10
      if (timestep > 10 && timestep <= 15) {
        llvm::errs() << "DEBUG: Found computational op " << timestep << ": "
                     << walkOp->getName() << "\n";
      }

      // Remove operands with single use (consumed by this operation)
      for (Value operand : walkOp->getOperands()) {
        if (operand.hasOneUse() &&
            isa<lwe::NewLWECiphertextType, lwe::LWECiphertextType>(
                operand.getType())) {
          liveCiphertexts.erase(operand);
        }
      }

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
        currentState.isKeyMap[ct] = false;
      }

      // For bootstrap operations, add all live keys
      if (isa<openfhe::BootstrapOp>(walkOp)) {
        for (auto &[rotIndex, towers] : liveKeys) {
          currentState.globalKeyTowers[rotIndex] = towers;
        }
      }

      // Record this timestep
      OperationLivenessInfo info = currentState.toOperationInfo(walkOp);
      results.addOperationInfo(info);

      // Debug output for first 20 timesteps and every 100th after that
      if (timestep <= 20 || timestep % 100 == 0) {
        llvm::errs() << "Timestep " << timestep << " (" << walkOp->getName()
                     << "): " << liveCiphertexts.size() << " live CTs, "
                     << liveKeys.size() << " live keys";
        if (isa<openfhe::BootstrapOp>(walkOp)) {
          llvm::errs() << " [BOOTSTRAP]";
        }
        llvm::errs() << "\n";
      }
    }
  });

  llvm::errs() << "=== Analysis Complete: " << timestep << " timesteps, "
               << totalOpsProcessed << " total operations processed ===\n";
  return results;
}

}  // namespace heir
}  // namespace mlir
