#include "lib/Dialect/KMRT/Transforms/RotationKeyLivenessDFA.h"

#include "lib/Dialect/KMRT/IR/KMRTOps.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project

#define DEBUG_TYPE "rotation-key-liveness-dfa"

namespace mlir {
namespace heir {
namespace kmrt {

// KeyIndex implementation
void KeyIndex::print(raw_ostream &os) const {
  if (isConstant()) {
    os << "const(" << *constantValue << ")";
  } else if (isSymbolic()) {
    os << "symbolic(" << symbolicValue << ")";
  } else {
    os << "unknown";
  }
}

// RotationKeyState implementation
RotationKeyState RotationKeyState::join(const RotationKeyState &lhs,
                                         const RotationKeyState &rhs) {
  // If either is unknown, take the other
  if (lhs.isUnknown()) return rhs;
  if (rhs.isUnknown()) return lhs;

  // If both have same key index and status, keep it
  if (lhs.keyIndex == rhs.keyIndex && lhs.status == rhs.status) {
    return lhs;
  }

  // Conservative join: if different states meet, mark as unknown
  // This means the key state depends on control flow
  return RotationKeyState();
}

void RotationKeyState::print(raw_ostream &os) const {
  os << "RotationKeyState(";
  switch (status) {
    case Status::Unknown:
      os << "unknown";
      break;
    case Status::NotLoaded:
      os << "not_loaded";
      break;
    case Status::Loaded:
      os << "loaded, index=";
      keyIndex.print(os);
      break;
    case Status::Cleared:
      os << "cleared, index=";
      keyIndex.print(os);
      break;
  }
  os << ")";
}

// Helper to extract key index from LoadKeyOp
static std::optional<KeyIndex> extractKeyIndex(LoadKeyOp loadOp) {
  Value indexValue = loadOp.getIndex();

  // Try to get constant index
  if (auto constOp = indexValue.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      return KeyIndex(intAttr.getInt());
    }
  }

  // Otherwise it's a symbolic index
  return KeyIndex(indexValue);
}

// RotationKeyLivenessDFA implementation
LogicalResult RotationKeyLivenessDFA::visitOperation(
    Operation *op, ArrayRef<const RotationKeyLattice *> operands,
    ArrayRef<RotationKeyLattice *> results) {

  LLVM_DEBUG(llvm::dbgs() << "RotationKeyLivenessDFA: visiting " << *op << "\n");

  // Helper to propagate state to all results
  auto propagateToResults = [&](const RotationKeyState &state) {
    for (auto *result : results) {
      ChangeResult changed = result->join(state);
      propagateIfChanged(result, changed);
    }
  };

  // Handle LoadKeyOp - creates a new live key
  if (auto loadOp = dyn_cast<LoadKeyOp>(op)) {
    auto maybeIndex = extractKeyIndex(loadOp);
    if (!maybeIndex) {
      LLVM_DEBUG(llvm::dbgs() << "  Could not extract key index\n");
      propagateToResults(RotationKeyState());
      return success();
    }

    RotationKeyState state(*maybeIndex, RotationKeyState::Status::Loaded);
    LLVM_DEBUG(llvm::dbgs() << "  LoadKeyOp: "; state.print(llvm::dbgs()); llvm::dbgs() << "\n");
    propagateToResults(state);
    return success();
  }

  // Handle ClearKeyOp - marks key as cleared
  if (auto clearOp = dyn_cast<ClearKeyOp>(op)) {
    if (!operands.empty() && operands[0]) {
      const auto &keyState = operands[0]->getValue();
      if (keyState.isLoaded()) {
        RotationKeyState clearedState(keyState.getKeyIndex(),
                                       RotationKeyState::Status::Cleared);
        LLVM_DEBUG(llvm::dbgs() << "  ClearKeyOp: "; clearedState.print(llvm::dbgs()); llvm::dbgs() << "\n");
        propagateToResults(clearedState);
        return success();
      }
    }
    propagateToResults(RotationKeyState());
    return success();
  }

  // Handle UseKeyOp - key remains live (passes through)
  if (auto useOp = dyn_cast<UseKeyOp>(op)) {
    if (!operands.empty() && operands[0]) {
      const auto &keyState = operands[0]->getValue();
      LLVM_DEBUG(llvm::dbgs() << "  UseKeyOp: "; keyState.print(llvm::dbgs()); llvm::dbgs() << "\n");
      propagateToResults(keyState);
      return success();
    }
    propagateToResults(RotationKeyState());
    return success();
  }

  // Default: propagate operand state to results
  if (!operands.empty() && operands[0] && !operands[0]->getValue().isUnknown()) {
    LLVM_DEBUG(llvm::dbgs() << "  Default propagation\n");
    propagateToResults(operands[0]->getValue());
  }

  return success();
}

void RotationKeyLivenessDFA::setToEntryState(RotationKeyLattice *lattice) {
  // Entry state: no keys loaded
  propagateIfChanged(lattice, lattice->join(RotationKeyState()));
}

}  // namespace kmrt
}  // namespace heir
}  // namespace mlir
