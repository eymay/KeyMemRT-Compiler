#ifndef LIB_ANALYSIS_KEYANALYSIS_KEYANALYSIS_H_
#define LIB_ANALYSIS_KEYANALYSIS_KEYANALYSIS_H_

#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lwe {

// State tracking the index of an evaluation key
class KeyState {
 public:
  KeyState() : index(std::nullopt) {}
  explicit KeyState(int64_t index) : index(index) {}

  bool isInitialized() const { return index.has_value(); }
  int64_t getIndex() const;
  bool operator==(const KeyState &rhs) const;
  static KeyState join(const KeyState &lhs, const KeyState &rhs);

  void print(raw_ostream &os) const;
  friend raw_ostream &operator<<(raw_ostream &os, const KeyState &state) {
    state.print(os);
    return os;
  }

 private:
  std::optional<int64_t> index;
};

// Dataflow lattice element for key state
class KeyStateLattice : public dataflow::Lattice<KeyState> {
 public:
  using Lattice::Lattice;
};

// Analysis to track evaluation key states
class KeyStateAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<KeyStateLattice> {
 public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const KeyStateLattice *> operands,
                               ArrayRef<KeyStateLattice *> results) override;

  void setToEntryState(KeyStateLattice *lattice) override;
};

}  // namespace lwe
}  // namespace heir
}  // namespace mlir

#endif  // LIB_ANALYSIS_KEYANALYSIS_KEYANALYSIS_H_
