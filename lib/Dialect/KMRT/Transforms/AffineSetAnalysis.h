#ifndef LIB_DIALECT_KMRT_TRANSFORMS_AFFINESETANALYSIS_H_
#define LIB_DIALECT_KMRT_TRANSFORMS_AFFINESETANALYSIS_H_

#include "mlir/include/mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace kmrt {

// Represents a set of key indices as an affine expression
// E.g., {i : lb <= i < ub} or {i*step : ...} or {constant}
struct AffineKeySet {
  // Constant key index (if this is a constant set)
  std::optional<int64_t> constantIndex;

  // For affine expressions: f(iv1, iv2, ...) where f is an affine map
  // E.g., iv * 4 + offset
  AffineExpr affineExpr;  // The expression in terms of IVs
  SmallVector<Value, 4> inductionVars;  // The IVs used in the expression

  // Bounds for each IV
  SmallVector<std::pair<int64_t, int64_t>, 4> ivBounds;  // [(lb, ub), ...]

  bool isConstant() const { return constantIndex.has_value(); }
  bool isAffine() const { return static_cast<bool>(affineExpr); }

  // Get the range of values this set can produce
  // Returns (min, max) or nullopt if unbounded/unknown
  std::optional<std::pair<int64_t, int64_t>> getRange() const;

  // Check if a constant value is in this set
  bool contains(int64_t value) const;

  // Check if two sets intersect
  bool intersects(const AffineKeySet &other) const;

  void print(raw_ostream &os) const;
};

// Analyzes a LoadKeyOp to extract its affine set representation
std::optional<AffineKeySet> extractAffineKeySet(Operation *loadOp);

// Analyzes a loop to extract the set of keys it loads
// Returns a vector of key sets, one for each LoadKeyOp in the loop
SmallVector<AffineKeySet> analyzeLoopKeySets(affine::AffineForOp loop);

// Determines if a preloop key (constant index) can be merged with a loop
// Returns the iteration indices where the key would be used
// E.g., key 5 merges with loop loading {i*4 : 0 <= i < 4} if:
//   - In first outer iteration (outer_iv == 0)
//   - When inner_iv == 5 / 4 = 1 (if 5 % 4 == 1)
struct MergeOpportunity {
  int64_t preloopKeyIndex;  // The constant key from before the loop
  SmallVector<std::pair<Value, int64_t>, 4> mergeConditions;  // [(iv, value), ...]

  bool isValid() const { return !mergeConditions.empty(); }
};

// Analyzes whether a preloop key can be merged with a nested loop structure
std::optional<MergeOpportunity> analyzeMergeOpportunity(
    int64_t preloopKeyIndex,
    affine::AffineForOp loop);

}  // namespace kmrt
}  // namespace heir
}  // namespace mlir

#endif  // LIB_DIALECT_KMRT_TRANSFORMS_AFFINESETANALYSIS_H_
