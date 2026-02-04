#include "lib/Dialect/KMRT/Transforms/AffineSetAnalysis.h"

#include "lib/Dialect/KMRT/IR/KMRTOps.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineExpr.h"  // from @llvm-project

#define DEBUG_TYPE "affine-set-analysis"

namespace mlir {
namespace heir {
namespace kmrt {

// AffineKeySet implementation

std::optional<std::pair<int64_t, int64_t>> AffineKeySet::getRange() const {
  if (isConstant()) {
    return std::make_pair(*constantIndex, *constantIndex);
  }

  if (!isAffine() || ivBounds.empty()) {
    return std::nullopt;
  }

  // For affine expressions, evaluate at bounds to get range
  // This is conservative - we evaluate at (lb, lb, ...) and (ub-1, ub-1, ...)
  SmallVector<int64_t> lbValues, ubValues;
  for (const auto &[lb, ub] : ivBounds) {
    lbValues.push_back(lb);
    ubValues.push_back(ub - 1);  // ub is exclusive
  }

  // Try to evaluate the affine expression
  // This is simplified - a full implementation would handle all affine operations
  if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(affineExpr)) {
    // Handle simple patterns like: d0 * c or d0 * s0
    if (binExpr.getKind() == AffineExprKind::Mul) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(binExpr.getLHS())) {
        if (auto constExpr = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
          // Pattern: d0 * c
          int64_t coeff = constExpr.getValue();
          int64_t minVal = lbValues[dimExpr.getPosition()] * coeff;
          int64_t maxVal = ubValues[dimExpr.getPosition()] * coeff;
          return std::make_pair(std::min(minVal, maxVal),
                                std::max(minVal, maxVal));
        }
      }
    } else if (binExpr.getKind() == AffineExprKind::Add) {
      // Pattern: expr1 + expr2
      // Recursively evaluate both sides
      // For now, simplified handling
    }
  } else if (auto dimExpr = dyn_cast<AffineDimExpr>(affineExpr)) {
    // Direct dimension: just the IV range
    return std::make_pair(lbValues[dimExpr.getPosition()],
                          ubValues[dimExpr.getPosition()]);
  } else if (auto constExpr = dyn_cast<AffineConstantExpr>(affineExpr)) {
    int64_t val = constExpr.getValue();
    return std::make_pair(val, val);
  }

  return std::nullopt;
}

bool AffineKeySet::contains(int64_t value) const {
  if (isConstant()) {
    return *constantIndex == value;
  }

  auto range = getRange();
  if (!range) return false;

  return value >= range->first && value <= range->second;
}

bool AffineKeySet::intersects(const AffineKeySet &other) const {
  // Both constant: direct comparison
  if (isConstant() && other.isConstant()) {
    return constantIndex == other.constantIndex;
  }

  // One constant, one affine: check if constant is in affine range
  if (isConstant()) {
    return other.contains(*constantIndex);
  }
  if (other.isConstant()) {
    return contains(*other.constantIndex);
  }

  // Both affine: check range overlap
  auto range1 = getRange();
  auto range2 = other.getRange();
  if (!range1 || !range2) return false;

  return !(range1->second < range2->first || range2->second < range1->first);
}

void AffineKeySet::print(raw_ostream &os) const {
  if (isConstant()) {
    os << "{" << *constantIndex << "}";
  } else if (isAffine()) {
    os << "{";
    affineExpr.print(os);
    os << " : ";
    for (size_t i = 0; i < ivBounds.size(); ++i) {
      if (i > 0) os << ", ";
      os << ivBounds[i].first << " <= iv" << i << " < " << ivBounds[i].second;
    }
    os << "}";
  } else {
    os << "{unknown}";
  }
}

// Helper to try extracting affine expression from a value
static std::optional<AffineExpr> tryExtractAffineExpr(
    Value val, SmallVectorImpl<Value> &ivs) {
  // Check if this is directly an IV
  for (size_t i = 0; i < ivs.size(); ++i) {
    if (val == ivs[i]) {
      return getAffineDimExpr(i, val.getContext());
    }
  }

  Operation *defOp = val.getDefiningOp();
  if (!defOp) return std::nullopt;

  // Handle index_cast
  if (auto castOp = dyn_cast<arith::IndexCastOp>(defOp)) {
    return tryExtractAffineExpr(castOp.getIn(), ivs);
  }

  // Handle affine.apply
  if (auto applyOp = dyn_cast<affine::AffineApplyOp>(defOp)) {
    AffineMap map = applyOp.getAffineMap();
    if (map.getNumResults() == 1) {
      // TODO: Map the operands to our IV list
      return map.getResult(0);
    }
  }

  // Handle arith.muli
  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    Value lhs = mulOp.getLhs();
    Value rhs = mulOp.getRhs();

    // Check for pattern: iv * constant
    auto lhsExpr = tryExtractAffineExpr(lhs, ivs);
    if (lhsExpr) {
      if (auto constOp = rhs.getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          MLIRContext *ctx = val.getContext();
          return *lhsExpr * getAffineConstantExpr(intAttr.getInt(), ctx);
        }
      }
    }

    // Check for pattern: constant * iv
    auto rhsExpr = tryExtractAffineExpr(rhs, ivs);
    if (rhsExpr) {
      if (auto constOp = lhs.getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          MLIRContext *ctx = val.getContext();
          return getAffineConstantExpr(intAttr.getInt(), ctx) * *rhsExpr;
        }
      }
    }
  }

  return std::nullopt;
}

std::optional<AffineKeySet> extractAffineKeySet(Operation *op) {
  auto loadOp = dyn_cast<LoadKeyOp>(op);
  if (!loadOp) return std::nullopt;

  Value indexValue = loadOp.getIndex();

  // Try constant index
  if (auto constOp = indexValue.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      AffineKeySet set;
      set.constantIndex = intAttr.getInt();
      return set;
    }
  }

  // Collect enclosing loop IVs
  SmallVector<Value, 4> ivs;
  SmallVector<std::pair<int64_t, int64_t>, 4> bounds;
  Operation *parent = op->getParentOp();
  while (parent) {
    if (auto loop = dyn_cast<affine::AffineForOp>(parent)) {
      ivs.push_back(loop.getInductionVar());
      auto lb = loop.getConstantLowerBound();
      auto ub = loop.getConstantUpperBound();
      bounds.push_back({lb, ub});
    }
    parent = parent->getParentOp();
  }

  if (ivs.empty()) return std::nullopt;

  // Try to extract affine expression
  auto maybeExpr = tryExtractAffineExpr(indexValue, ivs);
  if (!maybeExpr) return std::nullopt;

  AffineKeySet set;
  set.affineExpr = *maybeExpr;
  set.inductionVars = ivs;
  set.ivBounds = bounds;
  return set;
}

SmallVector<AffineKeySet> analyzeLoopKeySets(affine::AffineForOp loop) {
  SmallVector<AffineKeySet> sets;

  loop.walk([&](LoadKeyOp loadOp) {
    auto maybeSet = extractAffineKeySet(loadOp);
    if (maybeSet) {
      sets.push_back(*maybeSet);
      LLVM_DEBUG(llvm::dbgs() << "Extracted key set: ";
                 maybeSet->print(llvm::dbgs());
                 llvm::dbgs() << "\n");
    }
  });

  return sets;
}

std::optional<MergeOpportunity> analyzeMergeOpportunity(
    int64_t preloopKeyIndex, affine::AffineForOp loop) {

  LLVM_DEBUG(llvm::dbgs() << "Analyzing merge opportunity for preloop key "
                          << preloopKeyIndex << "\n");

  // Analyze all key sets loaded in the loop
  auto loopKeySets = analyzeLoopKeySets(loop);

  for (const auto &keySet : loopKeySets) {
    // Create a constant set for the preloop key
    AffineKeySet preloopSet;
    preloopSet.constantIndex = preloopKeyIndex;

    // Check intersection
    if (!keySet.intersects(preloopSet)) {
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "Found intersection with key set: ";
               keySet.print(llvm::dbgs());
               llvm::dbgs() << "\n");

    // Found an intersection! Now determine the merge conditions
    MergeOpportunity opp;
    opp.preloopKeyIndex = preloopKeyIndex;

    // For nested loops, we need to find which IV values produce the preloop key
    if (keySet.isAffine()) {
      // Try to solve: affineExpr(iv0, iv1, ...) == preloopKeyIndex
      // For simple cases like iv * step, we can compute: iv = preloopKeyIndex / step

      if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(keySet.affineExpr)) {
        if (binExpr.getKind() == AffineExprKind::Mul) {
          if (auto dimExpr = dyn_cast<AffineDimExpr>(binExpr.getLHS())) {
            if (auto constExpr = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
              // Pattern: d0 * c == preloopKeyIndex
              int64_t step = constExpr.getValue();
              if (preloopKeyIndex % step == 0) {
                int64_t ivValue = preloopKeyIndex / step;
                size_t ivPos = dimExpr.getPosition();

                // Check if ivValue is in bounds
                if (ivPos < keySet.ivBounds.size()) {
                  auto [lb, ub] = keySet.ivBounds[ivPos];
                  if (ivValue >= lb && ivValue < ub) {
                    opp.mergeConditions.push_back({keySet.inductionVars[ivPos], ivValue});
                    LLVM_DEBUG(llvm::dbgs() << "Merge condition: iv" << ivPos
                                            << " == " << ivValue << "\n");
                    return opp;
                  }
                }
              }
            }
          }
        }
      } else if (auto dimExpr = dyn_cast<AffineDimExpr>(keySet.affineExpr)) {
        // Direct IV: preloopKeyIndex must be within loop bounds
        size_t ivPos = dimExpr.getPosition();
        if (ivPos < keySet.ivBounds.size()) {
          auto [lb, ub] = keySet.ivBounds[ivPos];
          if (preloopKeyIndex >= lb && preloopKeyIndex < ub) {
            opp.mergeConditions.push_back({keySet.inductionVars[ivPos], preloopKeyIndex});
            LLVM_DEBUG(llvm::dbgs() << "Merge condition: iv" << ivPos
                                    << " == " << preloopKeyIndex << "\n");
            return opp;
          }
        }
      }
    }
  }

  return std::nullopt;
}

}  // namespace kmrt
}  // namespace heir
}  // namespace mlir
