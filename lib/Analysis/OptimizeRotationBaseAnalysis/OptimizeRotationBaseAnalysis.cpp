#include "lib/Analysis/OptimizeRotationBaseAnalysis/OptimizeRotationBaseAnalysis.h"

#include <cassert>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "llvm/include/llvm/ADT/STLExtras.h"
#include "llvm/include/llvm/Support/Debug.h"
#include "llvm/include/llvm/Support/raw_ostream.h"

// OR-Tools dependency
#include "ortools/math_opt/cpp/math_opt.h"

namespace math_opt = ::operations_research::math_opt;

namespace mlir {
namespace heir {

#define DEBUG_TYPE "optimize-rotation-base-analysis"

void RotationBaseAnalysis::collectRotationIndices() {
  llvm::DenseSet<int64_t> uniqueIndices;

  // Walk through the IR and collect all rotation indices from DeserializeKeyOp
  llvm::outs() << "Scanning for rotation indices...\n";
  opToRunOn->walk([&](openfhe::DeserializeKeyOp op) {
    if (auto indexAttr = op->getAttrOfType<IntegerAttr>("index")) {
      int64_t index = indexAttr.getInt();
      if (uniqueIndices.insert(index).second) {
        allRotationIndices.push_back(index);
        llvm::outs() << "Found rotation index: " << index << "\n";
      }
    }
  });

  llvm::outs() << "Total unique rotation indices found: "
               << allRotationIndices.size() << "\n";
}

LogicalResult RotationBaseAnalysis::solve() {
  // First collect all rotation indices from the IR
  collectRotationIndices();

  // If no indices found, return early
  if (allRotationIndices.empty()) {
    llvm::errs() << "No rotation indices found in the IR\n";
    return failure();
  }

  // If we have fewer indices than the base set size, just use all indices
  if (allRotationIndices.size() <= baseSetSize) {
    llvm::outs() << "Using all " << allRotationIndices.size()
                 << " indices as base set since it's <= requested size of "
                 << baseSetSize << "\n";
    optimalBaseSet = allRotationIndices;
    return success();
  }

  llvm::outs() << "Creating ILP model with base set size: " << baseSetSize
               << "\n";

  // Create the ILP model
  math_opt::Model model("RotationBaseOptimization");

  // Create variables for each potential base rotation index
  llvm::outs() << "Creating base variables...\n";
  std::vector<std::pair<int64_t, math_opt::Variable>> baseVars;
  for (int64_t index : allRotationIndices) {
    std::string varName = "Base_" + std::to_string(index);
    math_opt::Variable var = model.AddBinaryVariable(varName);
    baseVars.push_back({index, var});
  }

  // Create variables for representing each rotation as a sum of base rotations
  // For each rotation index i and base rotation b, use_b_for_i indicates how
  // many times base rotation b is used to compose rotation i
  llvm::outs() << "Creating representation variables...\n";

  // We can't use a 2D vector of Variables because Variable has no default
  // constructor Instead, use a flat vector with manual indexing
  std::vector<math_opt::Variable> useBaseVars;
  useBaseVars.reserve(allRotationIndices.size() * allRotationIndices.size());

  for (size_t targetIdx = 0; targetIdx < allRotationIndices.size();
       targetIdx++) {
    for (size_t baseIdx = 0; baseIdx < allRotationIndices.size(); baseIdx++) {
      int64_t targetIndex = allRotationIndices[targetIdx];
      int64_t baseIndex = allRotationIndices[baseIdx];

      std::string varName = "Use_" + std::to_string(baseIndex) + "_for_" +
                            std::to_string(targetIndex);

      // A base rotation can be used 0 to N times to compose a target rotation
      // Maximum is somewhat arbitrary - can set to the largest rotation index
      int maxUses = 20;  // Reasonable upper bound

      math_opt::Variable var = model.AddIntegerVariable(0, maxUses, varName);
      useBaseVars.push_back(var);
    }
  }

  // Helper function to find a base variable
  auto findBaseVar = [&baseVars](int64_t index) -> math_opt::Variable {
    for (const auto &pair : baseVars) {
      if (pair.first == index) {
        return pair.second;
      }
    }
    // Should never reach here
    assert(false && "Base variable not found");
    return baseVars[0].second;
  };

  // Helper function to find index in allRotationIndices
  auto findIndex = [this](int64_t index) -> size_t {
    for (size_t i = 0; i < this->allRotationIndices.size(); i++) {
      if (this->allRotationIndices[i] == index) {
        return i;
      }
    }
    // Should never reach here
    assert(false && "Index not found");
    return 0;
  };

  // Helper function to get variable from the flat vector
  auto getVar = [&useBaseVars, this](size_t targetIdx,
                                     size_t baseIdx) -> math_opt::Variable & {
    size_t flatIndex = targetIdx * this->allRotationIndices.size() + baseIdx;
    return useBaseVars[flatIndex];
  };

  // Constraint: The base set size must be exactly baseSetSize
  llvm::outs() << "Adding base set size constraint...\n";
  math_opt::LinearExpression baseSetSizeExpr;
  for (const auto &pair : baseVars) {
    baseSetSizeExpr += pair.second;
  }
  model.AddLinearConstraint(baseSetSizeExpr == baseSetSize, "BaseSetSize");

  // Constraint: Each rotation must be expressible as a sum of base rotations
  llvm::outs() << "Adding representation constraints...\n";
  for (size_t targetIdx = 0; targetIdx < allRotationIndices.size();
       targetIdx++) {
    int64_t targetIndex = allRotationIndices[targetIdx];

    // Linear expression representing the sum of base rotations times their
    // usage count
    math_opt::LinearExpression sumExpr;

    for (size_t baseIdx = 0; baseIdx < allRotationIndices.size(); baseIdx++) {
      int64_t baseIndex = allRotationIndices[baseIdx];
      // baseIndex * useCount
      sumExpr += baseIndex * getVar(targetIdx, baseIdx);
    }

    // The sum must equal the target rotation index
    std::string constraintName = "Compose_" + std::to_string(targetIndex);
    model.AddLinearConstraint(sumExpr == targetIndex, constraintName);
  }

  // Constraint: Can only use base rotations that are in the base set
  llvm::outs() << "Adding base set usage constraints...\n";
  for (size_t targetIdx = 0; targetIdx < allRotationIndices.size();
       targetIdx++) {
    for (size_t baseIdx = 0; baseIdx < allRotationIndices.size(); baseIdx++) {
      int64_t baseIndex = allRotationIndices[baseIdx];
      math_opt::Variable baseVar = findBaseVar(baseIndex);
      math_opt::Variable &useVar = getVar(targetIdx, baseIdx);

      // If baseIndex is not in the base set (baseVar = 0),
      // then useVar must be 0.
      // useVar <= maxUses * baseVar
      std::string constraintName = "OnlyUseBase_" + std::to_string(targetIdx) +
                                   "_" + std::to_string(baseIdx);

      int maxUses = 20;  // Same as before
      model.AddLinearConstraint(useVar <= maxUses * baseVar, constraintName);
    }
  }

  // Objective: Minimize the total number of base rotation operations
  llvm::outs() << "Setting objective function...\n";
  math_opt::LinearExpression objective;

  // For each target rotation, sum all uses of base rotations
  for (size_t targetIdx = 0; targetIdx < allRotationIndices.size();
       targetIdx++) {
    for (size_t baseIdx = 0; baseIdx < allRotationIndices.size(); baseIdx++) {
      objective += getVar(targetIdx, baseIdx);
    }
  }

  model.Minimize(objective);

  llvm::outs() << "Solving optimization problem...\n";

  // Solve the model
  const absl::StatusOr<math_opt::SolveResult> status =
      math_opt::Solve(model, math_opt::SolverType::kGscip);

  if (!status.ok()) {
    llvm::errs() << "Error solving the optimization problem\n";
    return failure();
  }

  const math_opt::SolveResult &result = status.value();

  switch (result.termination.reason) {
    case math_opt::TerminationReason::kOptimal:
      llvm::outs() << "Found optimal solution with objective value: "
                   << result.objective_value() << "\n";
      break;
    case math_opt::TerminationReason::kFeasible:
      llvm::outs()
          << "Found feasible (non-optimal) solution with objective value: "
          << result.objective_value() << "\n";
      break;
    default:
      llvm::errs() << "The problem does not have a feasible solution.\n";
      llvm::errs() << "Termination reason: "
                   << static_cast<int>(result.termination.reason) << "\n";

      // Try to provide more diagnostic information
      llvm::errs() << "This could be because:\n";
      llvm::errs() << "1. The base set size (" << baseSetSize
                   << ") is too small for the given rotation indices\n";
      llvm::errs()
          << "2. Some rotations can't be expressed as sums of others\n";

      return failure();
  }

  // Extract the solution
  auto varValues = result.variable_values();

  // Get the base set
  optimalBaseSet.clear();
  llvm::outs() << "Optimal base set: ";
  for (const auto &pair : baseVars) {
    if (varValues[pair.second] > 0.5) {  // Binary var should be close to 0 or 1
      optimalBaseSet.push_back(pair.first);
      llvm::outs() << pair.first << " ";
    }
  }
  llvm::outs() << "\n";

  // Build the composition mappings for each rotation
  compositionMappings.clear();
  llvm::outs() << "Rotation compositions:\n";

  for (size_t targetIdx = 0; targetIdx < allRotationIndices.size();
       targetIdx++) {
    int64_t targetIndex = allRotationIndices[targetIdx];

    // Skip base rotations
    if (isInBaseSet(targetIndex)) {
      llvm::outs() << targetIndex << " - Base rotation\n";
      continue;
    }

    llvm::outs() << targetIndex << " = ";
    bool firstTerm = true;

    for (int64_t baseIndex : optimalBaseSet) {
      size_t baseIdx = findIndex(baseIndex);
      int64_t useCount = std::round(varValues[getVar(targetIdx, baseIdx)]);

      if (useCount > 0) {
        if (!firstTerm) {
          llvm::outs() << " + ";
        }

        if (useCount == 1) {
          llvm::outs() << baseIndex;
        } else {
          llvm::outs() << useCount << "*" << baseIndex;
        }

        // Store the composition in our mapping
        compositionMappings[targetIndex][baseIndex] = useCount;

        firstTerm = false;
      }
    }
    llvm::outs() << " (total ops: ";

    // Calculate total operations
    int64_t totalOps = 0;
    for (const auto &entry : compositionMappings[targetIndex]) {
      totalOps += entry.second;
    }
    llvm::outs() << totalOps << ")\n";
  }

  // Calculate and display total operations needed
  int64_t grandTotal = 0;
  for (int64_t targetIndex : allRotationIndices) {
    if (isInBaseSet(targetIndex)) continue;

    int64_t targetTotal = 0;
    for (const auto &entry : compositionMappings[targetIndex]) {
      targetTotal += entry.second;
    }
    grandTotal += targetTotal;
  }

  llvm::outs() << "\nTotal operations for all non-base rotations: "
               << grandTotal << "\n";

  return success();
}

// Get the composition of a rotation index in terms of base rotations
std::map<int64_t, int64_t> RotationBaseAnalysis::getComposition(
    int64_t index) const {
  if (isInBaseSet(index)) {
    // If the index is in the base set, just return itself with count 1
    return {{index, 1}};
  }

  auto it = compositionMappings.find(index);
  if (it != compositionMappings.end()) {
    return it->second;
  }

  // If we can't find the composition, return empty map
  return {};
}

// Get the total number of operations needed for a rotation
int64_t RotationBaseAnalysis::getTotalOperations(int64_t index) const {
  if (isInBaseSet(index)) {
    return 1;  // Base rotation is just one operation
  }

  auto composition = getComposition(index);
  int64_t total = 0;
  for (const auto &entry : composition) {
    total += entry.second;
  }
  return total;
}

}  // namespace heir
}  // namespace mlir
