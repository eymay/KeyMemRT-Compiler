#ifndef LIB_DIALECT_KMRT_TRANSFORMS_ROTATIONKEYLIVENESSDFA_H_
#define LIB_DIALECT_KMRT_TRANSFORMS_ROTATIONKEYLIVENESSDFA_H_

#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/DenseMap.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace kmrt {

// Represents the index of a rotation key - either constant or symbolic
struct KeyIndex {
  // For constant indices
  std::optional<int64_t> constantValue;

  // For symbolic/dynamic indices - store the SSA value
  Value symbolicValue;

  KeyIndex() = default;
  explicit KeyIndex(int64_t constant) : constantValue(constant) {}
  explicit KeyIndex(Value symbolic) : symbolicValue(symbolic) {}

  bool isConstant() const { return constantValue.has_value(); }
  bool isSymbolic() const { return static_cast<bool>(symbolicValue); }

  bool operator==(const KeyIndex &other) const {
    if (isConstant() && other.isConstant()) {
      return constantValue == other.constantValue;
    }
    if (isSymbolic() && other.isSymbolic()) {
      return symbolicValue == other.symbolicValue;
    }
    return false;
  }

  void print(raw_ostream &os) const;
};

// State representing the liveness of a rotation key
class RotationKeyState {
 public:
  enum class Status {
    Unknown,      // State not yet determined
    NotLoaded,    // Key is not loaded
    Loaded,       // Key is loaded and live
    Cleared       // Key has been cleared
  };

  RotationKeyState() : status(Status::Unknown) {}

  explicit RotationKeyState(KeyIndex idx, Status s = Status::Loaded)
      : keyIndex(idx), status(s) {}

  bool isUnknown() const { return status == Status::Unknown; }
  bool isLoaded() const { return status == Status::Loaded; }
  bool isCleared() const { return status == Status::Cleared; }

  const KeyIndex &getKeyIndex() const { return keyIndex; }
  Status getStatus() const { return status; }

  // Join operation for dataflow lattice
  static RotationKeyState join(const RotationKeyState &lhs,
                                const RotationKeyState &rhs);

  bool operator==(const RotationKeyState &rhs) const {
    if (status != rhs.status) return false;
    if (status == Status::Unknown) return true;
    return keyIndex == rhs.keyIndex;
  }

  void print(raw_ostream &os) const;

 private:
  KeyIndex keyIndex;
  Status status;
};

// Dataflow lattice element for rotation key state
class RotationKeyLattice : public dataflow::Lattice<RotationKeyState> {
 public:
  using Lattice::Lattice;
};

// Sparse forward dataflow analysis for rotation key liveness
class RotationKeyLivenessDFA
    : public dataflow::SparseForwardDataFlowAnalysis<RotationKeyLattice> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const RotationKeyLattice *> operands,
                               ArrayRef<RotationKeyLattice *> results) override;

  void setToEntryState(RotationKeyLattice *lattice) override;
};

}  // namespace kmrt
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_KMRT_TRANSFORMS_ROTATIONKEYLIVENESSDFA_H_
