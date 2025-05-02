#include "lib/Analysis/OptimizeRotationBaseAnalysis/OptimizeRotationBaseAnalysis.h"

#include <algorithm>
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

int64_t RotationBaseAnalysis::computeGCD(int64_t a, int64_t b) {
  while (b != 0) {
    int64_t temp = b;
    b = a % b;
    a = temp;
  }
  return a;
}

std::vector<int64_t> RotationBaseAnalysis::generateSubsetSums(
    const std::vector<int64_t> &elements, int subsetSize, int64_t maxValue) {
  std::unordered_set<int64_t> sums;

  // Use a recursive helper function to generate combinations
  std::function<void(size_t, int, int64_t)> generateCombinations =
      [&](size_t startIdx, int remainingElements, int64_t currentSum) {
        // Base case: if we've selected the required number of elements
        if (remainingElements == 0) {
          sums.insert(currentSum);
          return;
        }

        // If we've reached the end of the array or sum exceeds max, return
        if (startIdx >= elements.size() || currentSum > maxValue) return;

        // Include the current element
        generateCombinations(startIdx + 1, remainingElements - 1,
                             currentSum + elements[startIdx]);

        // Exclude the current element
        generateCombinations(startIdx + 1, remainingElements, currentSum);
      };

  generateCombinations(0, subsetSize, 0);

  return std::vector<int64_t>(sums.begin(), sums.end());
}

void RotationBaseAnalysis::generateCandidateIndices() {
  // First collect all target indices from the IR
  collectRotationIndices();

  // Find the maximum rotation index for bounds
  int64_t maxIndex = 0;
  for (int64_t index : allRotationIndices) {
    maxIndex = std::max(maxIndex, index);
  }

  // Store candidates in a set to avoid duplicates
  std::set<int64_t> candidates;

  // 1. Add original indices
  for (int64_t index : allRotationIndices) {
    candidates.insert(index);
  }

  // 2. Add small integers that could be useful base elements
  // (Usually small rotations are efficient building blocks)
  int64_t smallIntLimit = std::min(static_cast<int64_t>(20), maxIndex);
  for (int64_t i = 1; i <= smallIntLimit; i++) {
    candidates.insert(i);
  }

  // 3. Add first-order differences (target - base = candidate)
  for (int64_t target : allRotationIndices) {
    for (int64_t base : allRotationIndices) {
      if (base < target) {
        int64_t diff = target - base;
        if (diff > 0 && diff != target && diff != base) {
          candidates.insert(diff);
        }
      }
    }
  }

  // 4. Add common pairwise differences
  for (size_t i = 0; i < allRotationIndices.size(); i++) {
    for (size_t j = i + 1; j < allRotationIndices.size(); j++) {
      int64_t diff = std::abs(allRotationIndices[i] - allRotationIndices[j]);
      if (diff > 0) {
        candidates.insert(diff);
      }
    }
  }

  // 5. Generate subset sums and higher-order differences
  const int MAX_SUBSET_SIZE = 2;  // Limit to avoid combinatorial explosion

  // Generate first layer of subset sums
  std::vector<int64_t> currentIndices(allRotationIndices.begin(),
                                      allRotationIndices.end());

  for (int subsetSize = 2; subsetSize <= MAX_SUBSET_SIZE; subsetSize++) {
    // Generate all subset sums of the current size
    std::vector<int64_t> subsetSums =
        generateSubsetSums(currentIndices, subsetSize, maxIndex);

    // Find higher-order differences
    for (int64_t target : allRotationIndices) {
      for (int64_t sum : subsetSums) {
        if (sum < target) {
          int64_t diff = target - sum;
          if (diff > 0 && diff < target) {
            candidates.insert(diff);
          }
        }
      }
    }
  }

  // 6. Look for "stepping stone" values to fill gaps in the sequence
  std::vector<int64_t> sortedIndices(allRotationIndices.begin(),
                                     allRotationIndices.end());
  std::sort(sortedIndices.begin(), sortedIndices.end());

  for (size_t i = 0; i < sortedIndices.size() - 1; i++) {
    int64_t current = sortedIndices[i];
    int64_t next = sortedIndices[i + 1];
    int64_t gap = next - current;

    // For large gaps, add potential midpoints
    if (gap > 3) {
      // Add midpoint
      candidates.insert(current + gap / 2);

      // For very large gaps, add quarter points
      if (gap > 8) {
        candidates.insert(current + gap / 4);
        candidates.insert(current + 3 * gap / 4);
      }
    }
  }

  // Limit the candidate set size to keep the ILP tractable
  const size_t MAX_CANDIDATES = 200;
  if (candidates.size() > MAX_CANDIDATES) {
    llvm::outs() << "Warning: Pruning candidate set from " << candidates.size()
                 << " to " << MAX_CANDIDATES << "\n";

    // Prioritize original indices and small values
    std::vector<int64_t> priorityCandidates;

    // Keep all original indices
    for (int64_t idx : allRotationIndices) {
      priorityCandidates.push_back(idx);
    }

    // Keep small integers
    for (int64_t i = 1; i <= smallIntLimit; i++) {
      if (candidates.count(i) > 0) {
        priorityCandidates.push_back(i);
      }
    }

    // Add other candidates sorted by value (prefer smaller values)
    std::vector<int64_t> remainingCandidates;
    for (int64_t c : candidates) {
      if (std::find(priorityCandidates.begin(), priorityCandidates.end(), c) ==
          priorityCandidates.end()) {
        remainingCandidates.push_back(c);
      }
    }

    std::sort(remainingCandidates.begin(), remainingCandidates.end());

    // Fill up to MAX_CANDIDATES
    size_t availableSlots = MAX_CANDIDATES - priorityCandidates.size();
    if (availableSlots > 0) {
      size_t numToAdd = std::min(availableSlots, remainingCandidates.size());
      priorityCandidates.insert(priorityCandidates.end(),
                                remainingCandidates.begin(),
                                remainingCandidates.begin() + numToAdd);
    }

    // Update candidates
    candidates.clear();
    for (int64_t c : priorityCandidates) {
      candidates.insert(c);
    }
  }

  // Convert to vector for the solver
  candidateIndices.clear();
  candidateIndices.insert(candidateIndices.end(), candidates.begin(),
                          candidates.end());

  llvm::outs() << "Generated " << candidateIndices.size()
               << " candidate indices for base set\n";
}

LogicalResult RotationBaseAnalysis::solve() {
  // Generate the expanded set of candidate indices
  generateCandidateIndices();

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
  for (int64_t index : candidateIndices) {
    std::string varName = "Base_" + std::to_string(index);
    math_opt::Variable var = model.AddBinaryVariable(varName);
    baseVars.push_back({index, var});
  }

  // Create variables for representing each rotation as a sum of base rotations
  // For each rotation index i and base rotation b, use_b_for_i indicates how
  // many times base rotation b is used to compose rotation i
  llvm::outs() << "Creating representation variables...\n";

  // We can't use a 2D vector of Variables because Variable has no default
  // constructor. Instead, use a flat vector with manual indexing
  std::vector<math_opt::Variable> useBaseVars;
  useBaseVars.reserve(allRotationIndices.size() * candidateIndices.size());

  for (size_t targetIdx = 0; targetIdx < allRotationIndices.size();
       targetIdx++) {
    for (size_t baseIdx = 0; baseIdx < candidateIndices.size(); baseIdx++) {
      int64_t targetIndex = allRotationIndices[targetIdx];
      int64_t baseIndex = candidateIndices[baseIdx];

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
  auto findTargetIndex = [this](int64_t index) -> size_t {
    for (size_t i = 0; i < this->allRotationIndices.size(); i++) {
      if (this->allRotationIndices[i] == index) {
        return i;
      }
    }
    // Should never reach here
    assert(false && "Target index not found");
    return 0;
  };

  // Helper function to find index in candidateIndices
  auto findCandidateIndex = [this](int64_t index) -> size_t {
    for (size_t i = 0; i < this->candidateIndices.size(); i++) {
      if (this->candidateIndices[i] == index) {
        return i;
      }
    }
    // Should never reach here
    assert(false && "Candidate index not found");
    return 0;
  };

  // Helper function to get variable from the flat vector
  auto getVar = [&useBaseVars, this](size_t targetIdx,
                                     size_t baseIdx) -> math_opt::Variable & {
    size_t flatIndex = targetIdx * this->candidateIndices.size() + baseIdx;
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

    for (size_t baseIdx = 0; baseIdx < candidateIndices.size(); baseIdx++) {
      int64_t baseIndex = candidateIndices[baseIdx];
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
    for (size_t baseIdx = 0; baseIdx < candidateIndices.size(); baseIdx++) {
      int64_t baseIndex = candidateIndices[baseIdx];
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
    for (size_t baseIdx = 0; baseIdx < candidateIndices.size(); baseIdx++) {
      objective += getVar(targetIdx, baseIdx);
    }
  }

  model.Minimize(objective);

  llvm::outs() << "Solving optimization problem...\n";

  math_opt::SolveArguments args;
  // Set the number of threads (e.g., 4 or 0 for automatic)
  args.parameters.threads = 32;
  // Solve the model
  const absl::StatusOr<math_opt::SolveResult> status =
      math_opt::Solve(model, math_opt::SolverType::kGscip, args);

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
      size_t baseIdx = findCandidateIndex(baseIndex);
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
    int64_t totalOps = getTotalOperations(targetIndex);
    llvm::outs() << totalOps << ")\n";
  }

  // Calculate and display total operations needed
  int64_t grandTotal = 0;
  for (int64_t targetIndex : allRotationIndices) {
    if (isInBaseSet(targetIndex)) continue;
    grandTotal += getTotalOperations(targetIndex);
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
