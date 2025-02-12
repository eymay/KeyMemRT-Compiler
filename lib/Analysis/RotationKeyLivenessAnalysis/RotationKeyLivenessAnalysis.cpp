#include "lib/Analysis/RotationKeyLivenessAnalysis/RotationKeyLivenessAnalysis.h"

#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"   // from @llvm-project

#define DEBUG_TYPE "key-analysis"

namespace mlir {
namespace heir {
namespace lwe {

// KeyState implementation
int64_t KeyState::getIndex() const {
  assert(isInitialized());
  return index.value();
}

bool KeyState::operator==(const KeyState &rhs) const {
  return index == rhs.index;
}

KeyState KeyState::join(const KeyState &lhs, const KeyState &rhs) {
  if (!lhs.isInitialized()) return rhs;
  if (!rhs.isInitialized()) return lhs;
  if (lhs.getIndex() != rhs.getIndex()) return KeyState();
  return lhs;
}

void KeyState::print(raw_ostream &os) const {
  if (isInitialized()) {
    os << "KeyState(" << index.value() << ")";
  } else {
    os << "KeyState(uninitialized)";
  }
}

// KeyStateAnalysis implementation
LogicalResult KeyStateAnalysis::visitOperation(
    Operation *op, ArrayRef<const KeyStateLattice *> operands,
    ArrayRef<KeyStateLattice *> results) {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing op: " << *op << "\n");

  // Utility function to propagate a key state to all results
  auto propagateToResults = [&](const KeyState &state) {
    for (auto *result : results) {
      ChangeResult changed = result->join(state);
      propagateIfChanged(result, changed);
    }
  };

  return llvm::TypeSwitch<Operation &, LogicalResult>(*op)
      // Handle deserialize key op - initializes key state
      .Case<openfhe::DeserializeKeyOp>([&](auto deserOp) {
        auto indexAttr =
            deserOp.getEvalKey().getType().getRotationIndex().getInt();
        propagateToResults(KeyState(indexAttr));
        return success();
      })

      // Handle key clearing - terminates key state
      .Case<openfhe::ClearKeyOp>([&](auto clearOp) {
        // When a key is cleared, its state should be nullified
        const auto *keyLattice = operands[0];
        if (keyLattice && keyLattice->getValue().isInitialized()) {
          // Set the key's state to cleared (uninitialized)
          propagateToResults(KeyState());
        }
        return success();
      })

      // Other ops just propagate key state
      .Default([&](Operation &) {
        // For non-key ops, propagate operand states to results
        if (!operands.empty() && operands[0]->getValue().isInitialized()) {
          propagateToResults(operands[0]->getValue());
        }
        return success();
      });
}

void KeyStateAnalysis::setToEntryState(KeyStateLattice *lattice) {
  propagateIfChanged(lattice, lattice->join(KeyState()));
}

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
