//===- PatternLoopRolling.cpp - Linear Transform Pattern Rolling ----------===//

#include "lib/Transforms/PatternLoopRolling/PatternLoopRolling.h"

#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "llvm/include/llvm/ADT/DenseSet.h"
#include "llvm/include/llvm/Support/Debug.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/Support/LLVM.h"

#define DEBUG_TYPE "pattern-loop-rolling"

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_PATTERNLOOPROLLING
#include "lib/Transforms/PatternLoopRolling/PatternLoopRolling.h.inc"

struct LinearTransformPattern {
  SmallVector<openfhe::DeserializeKeyOp, 8> deserializeOps;
  SmallVector<openfhe::RotOp, 8> rotOps;
  SmallVector<openfhe::MulPlainOp, 8> mulPlainOps;
  SmallVector<openfhe::AddOp, 8> addOps;
  SmallVector<Operation*, 16> clearOps;
  SmallVector<openfhe::EnqueueKeyOp, 8> enqueueOps;

  Value baseInput;
  Value accumulator;
  SmallVector<Value, 8> plaintexts;
  SmallVector<int64_t, 8> rotationIndices;

  // New: Track irregular patterns and exceptions
  struct PatternException {
    int loopIndex;           // Which loop iteration
    Value alternativeInput;  // Different base input for this iteration
    openfhe::DeserializeKeyOp
        altDeserialize;         // Alternative deserialize location
    bool skipMulPlain = false;  // Skip mul_plain for this iteration
    bool skipAdd = false;       // Skip add for this iteration
  };
  SmallVector<PatternException, 4> exceptions;

  // Track different base inputs used in the pattern
  SmallVector<Value, 4> baseInputs;  // Multiple base inputs
  SmallVector<SmallVector<int, 4>, 4>
      inputToIterations;  // Which iterations use which base

  bool isValid() const {
    // More flexible validation - allow irregular patterns
    bool basicStructure =
        deserializeOps.size() >= 2 && deserializeOps.size() == rotOps.size();

    // Allow flexible mul_plain and add operations
    bool flexibleStructure = mulPlainOps.size() <= deserializeOps.size() &&
                             addOps.size() <= deserializeOps.size();

    // Must have at least one base input
    bool hasBaseInput = !baseInputs.empty();

    return basicStructure && flexibleStructure && hasBaseInput;
  }

  int calculateSavings() const {
    if (!isValid()) return 0;

    int originalOps = deserializeOps.size() + rotOps.size() +
                      mulPlainOps.size() + addOps.size() + clearOps.size() +
                      enqueueOps.size();

    // Loop overhead scales with complexity
    int baseLoopOps = 10;                             // Basic loop setup
    int conditionalOverhead = exceptions.size() * 3;  // Cost of conditionals
    int inputSelectionCost = baseInputs.size() > 1 ? baseInputs.size() * 2 : 0;

    int loopOps = baseLoopOps + conditionalOverhead + inputSelectionCost;

    return std::max(0, originalOps - loopOps);
  }
};

struct PatternLoopRolling : impl::PatternLoopRollingBase<PatternLoopRolling> {
  using impl::PatternLoopRollingBase<
      PatternLoopRolling>::PatternLoopRollingBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    LLVM_DEBUG(llvm::dbgs()
               << "=== Enhanced PatternLoopRolling Pass Start ===\n");
    LLVM_DEBUG(llvm::dbgs() << "Function: " << func.getName() << "\n");

    // Skip helper functions
    if (func.getName().contains("__generate_crypto_context") ||
        func.getName().contains("__configure_crypto_context")) {
      return;
    }

    // Find linear transform patterns
    auto patterns = findLinearTransformPatterns(func);

    if (patterns.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No linear transform patterns found\n");
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "Found " << patterns.size()
                            << " linear transform patterns\n");

    // Transform patterns one at a time to avoid conflicts
    MLIRContext* context = &getContext();
    OpBuilder builder(context);
    int transformedPatterns = 0;

    // Sort patterns by savings (descending)
    std::sort(patterns.begin(), patterns.end(),
              [](const auto& a, const auto& b) {
                return a.calculateSavings() > b.calculateSavings();
              });

    // Transform patterns one by one, re-validating after each transformation
    for (int i = 0; i < patterns.size() && transformedPatterns < 5;
         ++i) {  // Limit to 5 patterns to be safe
      auto& pattern = patterns[i];

      // Re-validate pattern before transformation
      if (!isPatternStillValid(pattern)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Pattern " << i << " is no longer valid, skipping\n");
        continue;
      }

      if (pattern.calculateSavings() >=
          5) {  // Lower threshold for flexible patterns
        LLVM_DEBUG(llvm::dbgs()
                   << "Attempting to transform pattern " << i << " with "
                   << pattern.deserializeOps.size() << " iterations\n");

        if (succeeded(
                transformFlexibleLinearTransformPattern(builder, pattern))) {
          transformedPatterns++;
          LLVM_DEBUG(llvm::dbgs()
                     << "Successfully transformed flexible pattern with "
                     << pattern.deserializeOps.size() << " iterations, saved "
                     << pattern.calculateSavings() << " operations\n");

          // After transformation, invalidate any remaining patterns that might
          // conflict
          invalidateConflictingPatterns(patterns, i + 1, pattern);
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "Failed to transform pattern " << i << "\n");
        }
      }
    }

    LLVM_DEBUG(llvm::dbgs()
               << "=== Enhanced PatternLoopRolling Pass Complete: "
               << transformedPatterns << " patterns transformed ===\n");
  }

 private:
  SmallVector<LinearTransformPattern, 4> findLinearTransformPatterns(
      func::FuncOp func) {
    SmallVector<LinearTransformPattern, 4> patterns;

    // Collect operations in order
    SmallVector<Operation*, 256> operations;
    func.walk([&](Operation* op) {
      if (!isa<func::FuncOp, func::ReturnOp>(op)) {
        operations.push_back(op);
      }
    });

    LLVM_DEBUG(llvm::dbgs()
               << "Analyzing " << operations.size()
               << " operations for flexible linear transform patterns\n");

    // Try both regular and flexible pattern detection
    for (int start = 0; start < operations.size() - 5; ++start) {
      if (auto deserOp =
              dyn_cast<openfhe::DeserializeKeyOp>(operations[start])) {
        // Try flexible pattern detection first
        auto flexiblePattern =
            extractFlexibleLinearTransformPattern(operations, start);
        if (flexiblePattern.isValid()) {
          patterns.push_back(flexiblePattern);
          LLVM_DEBUG(llvm::dbgs()
                     << "Found flexible linear transform pattern with "
                     << flexiblePattern.deserializeOps.size()
                     << " iterations starting at " << start << "\n");
          continue;
        }

        // Fall back to regular pattern detection
        auto regularPattern = extractLinearTransformPattern(operations, start);
        if (regularPattern.isValid()) {
          patterns.push_back(regularPattern);
          LLVM_DEBUG(llvm::dbgs()
                     << "Found regular linear transform pattern with "
                     << regularPattern.deserializeOps.size()
                     << " iterations starting at " << start << "\n");
        }
      }
    }

    // Remove overlapping patterns (keep the one with more savings)
    llvm::DenseSet<Operation*> usedOps;
    auto end = std::remove_if(
        patterns.begin(), patterns.end(),
        [&](const LinearTransformPattern& pattern) {
          // Check if any operations are already used
          for (auto deserOp : pattern.deserializeOps) {
            if (usedOps.contains(const_cast<openfhe::DeserializeKeyOp&>(deserOp)
                                     .getOperation())) {
              LLVM_DEBUG(llvm::dbgs() << "Removing overlapping pattern\n");
              return true;  // Remove this pattern
            }
          }
          for (auto rotOp : pattern.rotOps) {
            if (usedOps.contains(
                    const_cast<openfhe::RotOp&>(rotOp).getOperation())) {
              LLVM_DEBUG(llvm::dbgs()
                         << "Removing overlapping pattern (rot conflict)\n");
              return true;  // Remove this pattern
            }
          }
          // Mark operations as used
          for (auto deserOp : pattern.deserializeOps) {
            usedOps.insert(
                const_cast<openfhe::DeserializeKeyOp&>(deserOp).getOperation());
          }
          for (auto rotOp : pattern.rotOps) {
            usedOps.insert(const_cast<openfhe::RotOp&>(rotOp).getOperation());
          }
          return false;  // Keep this pattern
        });
    patterns.erase(end, patterns.end());

    // Sort by savings (largest first)
    llvm::sort(patterns, [](const LinearTransformPattern& a,
                            const LinearTransformPattern& b) {
      return a.calculateSavings() > b.calculateSavings();
    });

    return patterns;
  }

  // Enhanced pattern extraction that can handle irregular sequences
  LinearTransformPattern extractFlexibleLinearTransformPattern(
      ArrayRef<Operation*> operations, int start) {
    LinearTransformPattern pattern;

    // First pass: collect all deserialize operations that might be part of the
    // pattern
    SmallVector<std::pair<openfhe::DeserializeKeyOp, int>, 8> deserializeOps;

    for (int i = start; i < std::min(start + 100, (int)operations.size());
         ++i) {
      if (auto deserOp = dyn_cast<openfhe::DeserializeKeyOp>(operations[i])) {
        deserializeOps.push_back({deserOp, i});
        if (deserializeOps.size() >= 8) break;  // Limit search
      }
    }

    if (deserializeOps.size() < 2) {
      return pattern;  // Need at least 2 operations
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Found " << deserializeOps.size()
               << " deserialize operations for flexible pattern\n");

    // Second pass: for each deserialize, find its corresponding rot, mul_plain,
    // add
    llvm::DenseMap<Value, int> baseInputToIndex;  // Track different base inputs
    int currentBaseIndex = 0;

    for (int iter = 0; iter < deserializeOps.size(); ++iter) {
      auto [deserOp, deserPos] = deserializeOps[iter];

      pattern.deserializeOps.push_back(deserOp);
      if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
        pattern.rotationIndices.push_back(indexAttr.getInt());
      }

      // Find corresponding rot operation
      openfhe::RotOp rotOp = nullptr;
      for (int j = deserPos + 1;
           j < std::min(deserPos + 30, (int)operations.size()); ++j) {
        if (auto candidateRot = dyn_cast<openfhe::RotOp>(operations[j])) {
          if (candidateRot.getEvalKey() == deserOp.getResult()) {
            rotOp = candidateRot;
            break;
          }
        }
      }

      if (!rotOp) {
        LLVM_DEBUG(llvm::dbgs()
                   << "No rot found for deserialize " << iter << "\n");
        // Remove this incomplete iteration
        pattern.deserializeOps.pop_back();
        if (!pattern.rotationIndices.empty()) {
          pattern.rotationIndices.pop_back();
        }
        continue;
      }

      pattern.rotOps.push_back(rotOp);

      // Track base input for this iteration
      Value rotInput = rotOp.getCiphertext();
      int baseIndex;
      if (baseInputToIndex.find(rotInput) == baseInputToIndex.end()) {
        baseIndex = currentBaseIndex++;
        baseInputToIndex[rotInput] = baseIndex;
        pattern.baseInputs.push_back(rotInput);
        pattern.inputToIterations.push_back(SmallVector<int, 4>());
      } else {
        baseIndex = baseInputToIndex[rotInput];
      }

      if (baseIndex < pattern.inputToIterations.size()) {
        pattern.inputToIterations[baseIndex].push_back(iter);
      }

      // Set primary base input as the most common one
      if (iter == 0) {
        pattern.baseInput = rotInput;
      }

      // Find corresponding mul_plain operation
      openfhe::MulPlainOp mulOp = nullptr;
      int mulPos = -1;
      for (int j = deserPos + 1;
           j < std::min(deserPos + 40, (int)operations.size()); ++j) {
        if (auto candidateMul = dyn_cast<openfhe::MulPlainOp>(operations[j])) {
          if (candidateMul.getCiphertext() == rotOp.getResult()) {
            mulOp = candidateMul;
            mulPos = j;
            break;
          }
        }
      }

      if (mulOp) {
        pattern.mulPlainOps.push_back(mulOp);
        pattern.plaintexts.push_back(mulOp.getPlaintext());

        // Find corresponding add operation
        openfhe::AddOp addOp = nullptr;
        for (int j = mulPos + 1;
             j < std::min(mulPos + 30, (int)operations.size()); ++j) {
          if (auto candidateAdd = dyn_cast<openfhe::AddOp>(operations[j])) {
            if (candidateAdd.getRhs() == mulOp.getResult() ||
                candidateAdd.getLhs() == mulOp.getResult()) {
              addOp = candidateAdd;
              break;
            }
          }
        }

        if (addOp) {
          pattern.addOps.push_back(addOp);
          if (!pattern.accumulator) {
            pattern.accumulator = (addOp.getRhs() == mulOp.getResult())
                                      ? addOp.getLhs()
                                      : addOp.getRhs();
          }
        } else {
          // Mark as exception - has mul_plain but no add
          pattern.addOps.push_back(openfhe::AddOp{});  // placeholder
          LinearTransformPattern::PatternException exc;
          exc.loopIndex = iter;
          exc.skipAdd = true;
          pattern.exceptions.push_back(exc);
        }
      } else {
        // Mark as exception - no mul_plain found
        pattern.mulPlainOps.push_back(openfhe::MulPlainOp{});  // placeholder
        pattern.plaintexts.push_back(Value{});                 // placeholder
        pattern.addOps.push_back(openfhe::AddOp{});            // placeholder

        LinearTransformPattern::PatternException exc;
        exc.loopIndex = iter;
        exc.skipMulPlain = true;
        exc.skipAdd = true;
        pattern.exceptions.push_back(exc);
      }

      LLVM_DEBUG(llvm::dbgs() << "Flexible pattern iteration " << iter
                              << " using base input " << baseIndex << "\n");
    }

    return pattern;
  }

  // Original pattern extraction for regular patterns
  LinearTransformPattern extractLinearTransformPattern(
      ArrayRef<Operation*> operations, int start) {
    LinearTransformPattern pattern;

    int currentPos = start;
    Value baseInput = nullptr;

    LLVM_DEBUG(llvm::dbgs() << "Starting regular pattern detection at position "
                            << start << "\n");

    while (currentPos < operations.size()) {
      // 1. Look for deserialize_key operation
      if (currentPos >= operations.size() ||
          !isa<openfhe::DeserializeKeyOp>(operations[currentPos])) {
        break;
      }

      auto deserOp = cast<openfhe::DeserializeKeyOp>(operations[currentPos]);
      pattern.deserializeOps.push_back(deserOp);

      // Extract rotation index
      if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
        pattern.rotationIndices.push_back(indexAttr.getInt());
      }

      currentPos++;

      // 2. Skip optional enqueue operations
      while (currentPos < operations.size() &&
             isa<openfhe::EnqueueKeyOp>(operations[currentPos])) {
        pattern.enqueueOps.push_back(
            cast<openfhe::EnqueueKeyOp>(operations[currentPos]));
        currentPos++;
      }

      // 3. Look for rot operation that uses the deserialized key
      openfhe::RotOp rotOp = nullptr;
      for (int lookahead = 0;
           lookahead < 10 && currentPos + lookahead < operations.size();
           ++lookahead) {
        if (auto candidateRot =
                dyn_cast<openfhe::RotOp>(operations[currentPos + lookahead])) {
          if (candidateRot.getEvalKey() == deserOp.getResult()) {
            rotOp = candidateRot;
            currentPos += lookahead;
            break;
          }
        }
      }

      if (!rotOp) {
        LLVM_DEBUG(llvm::dbgs() << "No matching rot found for deserialize\n");
        break;
      }

      pattern.rotOps.push_back(rotOp);

      // Check that all rotations use the same base input
      if (!baseInput) {
        baseInput = rotOp.getCiphertext();
        pattern.baseInput = baseInput;
        pattern.baseInputs.push_back(baseInput);
        pattern.inputToIterations.push_back(SmallVector<int, 4>());
      } else if (rotOp.getCiphertext() != baseInput) {
        LLVM_DEBUG(llvm::dbgs() << "Different base inputs, pattern broken\n");
        break;
      }

      currentPos++;

      // 4. Skip clear_key operations
      while (currentPos < operations.size() &&
             isa<openfhe::ClearKeyOp>(operations[currentPos])) {
        pattern.clearOps.push_back(operations[currentPos]);
        currentPos++;
      }

      // 5. Look for mul_plain that uses the rotated result
      openfhe::MulPlainOp mulOp = nullptr;
      for (int lookahead = 0;
           lookahead < 5 && currentPos + lookahead < operations.size();
           ++lookahead) {
        if (auto candidateMul = dyn_cast<openfhe::MulPlainOp>(
                operations[currentPos + lookahead])) {
          if (candidateMul.getCiphertext() == rotOp.getResult()) {
            mulOp = candidateMul;
            currentPos += lookahead;
            break;
          }
        }
      }

      if (!mulOp) {
        LLVM_DEBUG(llvm::dbgs() << "No matching mul_plain found for rot\n");
        break;
      }

      pattern.mulPlainOps.push_back(mulOp);
      pattern.plaintexts.push_back(mulOp.getPlaintext());
      currentPos++;

      // 6. Skip clear_ct operations
      while (currentPos < operations.size() &&
             isa<openfhe::ClearCtOp>(operations[currentPos])) {
        pattern.clearOps.push_back(operations[currentPos]);
        currentPos++;
      }

      // 7. Look for add operation that uses the mul_plain result
      openfhe::AddOp addOp = nullptr;
      for (int lookahead = 0;
           lookahead < 5 && currentPos + lookahead < operations.size();
           ++lookahead) {
        if (auto candidateAdd =
                dyn_cast<openfhe::AddOp>(operations[currentPos + lookahead])) {
          if (candidateAdd.getRhs() == mulOp.getResult() ||
              candidateAdd.getLhs() == mulOp.getResult()) {
            addOp = candidateAdd;
            currentPos += lookahead;
            break;
          }
        }
      }

      if (!addOp) {
        LLVM_DEBUG(llvm::dbgs() << "No matching add found for mul_plain\n");
        break;
      }

      pattern.addOps.push_back(addOp);
      if (!pattern.accumulator) {
        pattern.accumulator = (addOp.getRhs() == mulOp.getResult())
                                  ? addOp.getLhs()
                                  : addOp.getRhs();
      }

      currentPos++;

      // 8. Skip more clear_ct operations
      while (currentPos < operations.size() &&
             isa<openfhe::ClearCtOp>(operations[currentPos])) {
        pattern.clearOps.push_back(operations[currentPos]);
        currentPos++;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "Found iteration " << pattern.deserializeOps.size()
                 << " of regular linear transform pattern\n");
    }

    return pattern;
  }

  // Enhanced transformation that generates conditional code
  LogicalResult transformFlexibleLinearTransformPattern(
      OpBuilder& builder, const LinearTransformPattern& pattern) {
    if (!pattern.isValid()) {
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Transforming flexible pattern with "
                            << pattern.deserializeOps.size() << " iterations, "
                            << pattern.baseInputs.size() << " base inputs, "
                            << pattern.exceptions.size() << " exceptions\n");

    // Validate pattern before transformation
    for (int i = 0; i < pattern.deserializeOps.size(); ++i) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Iteration " << i << ": deserialize="
                 << const_cast<openfhe::DeserializeKeyOp&>(
                        pattern.deserializeOps[i])
                        .getOperation()
                 << " rot="
                 << (i < pattern.rotOps.size()
                         ? const_cast<openfhe::RotOp&>(pattern.rotOps[i])
                               .getOperation()
                         : nullptr)
                 << "\n");
    }

    Operation* firstOp =
        const_cast<openfhe::DeserializeKeyOp&>(pattern.deserializeOps[0])
            .getOperation();
    Location loc = firstOp->getLoc();

    // Hoist enqueue operations
    builder.setInsertionPoint(firstOp);
    for (auto enqueueOp : pattern.enqueueOps) {
      auto indexAttr = enqueueOp->getAttrOfType<IntegerAttr>("index");
      if (!indexAttr) continue;

      auto depthAttr = enqueueOp->getAttrOfType<IntegerAttr>("depth");
      builder.create<openfhe::EnqueueKeyOp>(loc, enqueueOp.getCryptoContext(),
                                            indexAttr, depthAttr);
    }

    // Create loop
    Value lowerBound = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value upperBound = builder.create<arith::ConstantIndexOp>(
        loc, pattern.deserializeOps.size());
    Value step = builder.create<arith::ConstantIndexOp>(loc, 1);

    SmallVector<Value> iterArgs;
    if (pattern.accumulator) {
      iterArgs.push_back(pattern.accumulator);
    }

    auto forOp =
        builder.create<scf::ForOp>(loc, lowerBound, upperBound, step, iterArgs);

    // Build flexible loop body
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(forOp.getBody());

    Value inductionVar = forOp.getInductionVar();
    Value currentAcc = iterArgs.empty() ? Value{} : forOp.getRegionIterArg(0);

    // Generate key selection with conditionals
    Value selectedKey =
        createFlexibleKeySelector(builder, loc, inductionVar, pattern);

    // Generate base input selection (if multiple base inputs)
    Value selectedInput;
    if (pattern.baseInputs.size() == 1) {
      selectedInput = pattern.baseInputs[0];
    } else {
      selectedInput =
          createBaseInputSelector(builder, loc, inductionVar, pattern);
    }

    // Rotation
    Value rotResult = builder.create<openfhe::RotOp>(
        loc,
        const_cast<openfhe::RotOp&>(pattern.rotOps[0]).getResult().getType(),
        const_cast<openfhe::RotOp&>(pattern.rotOps[0]).getCryptoContext(),
        selectedInput, selectedKey);

    // Clear key
    builder.create<openfhe::ClearKeyOp>(
        loc,
        const_cast<openfhe::DeserializeKeyOp&>(pattern.deserializeOps[0])
            .getCryptoContext(),
        selectedKey);

    Value finalResult = rotResult;

    // Conditional mul_plain and add operations
    if (!pattern.mulPlainOps.empty()) {
      finalResult = createConditionalMulPlainAndAdd(
          builder, loc, inductionVar, rotResult, currentAcc, pattern);
    } else if (currentAcc) {
      // Just accumulate the rotation result
      finalResult = builder.create<openfhe::AddOp>(
          loc, currentAcc.getType(),
          const_cast<openfhe::RotOp&>(pattern.rotOps[0]).getCryptoContext(),
          currentAcc, rotResult);
    }

    // Yield result
    SmallVector<Value> yieldValues;
    if (currentAcc) {
      yieldValues.push_back(finalResult);
    }
    builder.create<scf::YieldOp>(loc, yieldValues);

    // Replace uses and cleanup
    if (!iterArgs.empty()) {
      Value loopResult = forOp.getResult(0);

      // Find the final operation whose result should be replaced by the loop
      // result
      Operation* finalOp = nullptr;

      // Look for the last add operation that produces the final result
      for (int i = pattern.addOps.size() - 1; i >= 0; --i) {
        if (pattern.addOps[i]) {  // Valid add operation
          finalOp =
              const_cast<openfhe::AddOp&>(pattern.addOps[i]).getOperation();
          break;
        }
      }

      // If no add operations, look for the last mul_plain operation
      if (!finalOp) {
        for (int i = pattern.mulPlainOps.size() - 1; i >= 0; --i) {
          if (pattern.mulPlainOps[i]) {  // Valid mul_plain operation
            finalOp = const_cast<openfhe::MulPlainOp&>(pattern.mulPlainOps[i])
                          .getOperation();
            break;
          }
        }
      }

      // If no mul_plain operations, look for the last rot operation
      if (!finalOp) {
        for (int i = pattern.rotOps.size() - 1; i >= 0; --i) {
          finalOp =
              const_cast<openfhe::RotOp&>(pattern.rotOps[i]).getOperation();
          if (finalOp) break;
        }
      }

      // Replace all uses of the final operation's result with the loop result
      if (finalOp && finalOp->getNumResults() > 0) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "Replacing uses of final operation result with loop result\n");
        finalOp->getResult(0).replaceAllUsesWith(loopResult);
      }
    }

    cleanupLinearTransformPattern(pattern);
    return success();
  }

  // Create key selector with switch/if statements for irregular patterns
  Value createFlexibleKeySelector(OpBuilder& builder, Location loc,
                                  Value inductionVar,
                                  const LinearTransformPattern& pattern) {
    if (pattern.deserializeOps.size() == 1) {
      return const_cast<openfhe::DeserializeKeyOp&>(pattern.deserializeOps[0])
          .getResult();
    }

    // Create switch-like structure
    Value result =
        const_cast<openfhe::DeserializeKeyOp&>(pattern.deserializeOps[0])
            .getResult();

    for (int i = 1; i < pattern.deserializeOps.size(); ++i) {
      Value expectedIndex = builder.create<arith::ConstantIndexOp>(loc, i);
      Value condition = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, inductionVar, expectedIndex);

      auto ifOp = builder.create<scf::IfOp>(
          loc,
          const_cast<openfhe::DeserializeKeyOp&>(pattern.deserializeOps[i])
              .getResult()
              .getType(),
          condition, /*hasElse=*/true);

      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        builder.create<scf::YieldOp>(
            loc,
            const_cast<openfhe::DeserializeKeyOp&>(pattern.deserializeOps[i])
                .getResult());
      }

      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        builder.create<scf::YieldOp>(loc, result);
      }

      result = ifOp.getResult(0);
    }

    return result;
  }

  // Create base input selector for multiple base inputs
  Value createBaseInputSelector(OpBuilder& builder, Location loc,
                                Value inductionVar,
                                const LinearTransformPattern& pattern) {
    Value result = pattern.baseInputs[0];

    for (int baseIdx = 1; baseIdx < pattern.baseInputs.size(); ++baseIdx) {
      // Create condition for iterations that use this base input
      Value baseCondition;
      bool first = true;

      if (baseIdx < pattern.inputToIterations.size()) {
        for (int iter : pattern.inputToIterations[baseIdx]) {
          Value iterConstant =
              builder.create<arith::ConstantIndexOp>(loc, iter);
          Value iterCondition = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, inductionVar, iterConstant);

          if (first) {
            baseCondition = iterCondition;
            first = false;
          } else {
            baseCondition =
                builder.create<arith::OrIOp>(loc, baseCondition, iterCondition);
          }
        }
      }

      if (baseCondition) {
        auto ifOp = builder.create<scf::IfOp>(
            loc, pattern.baseInputs[baseIdx].getType(), baseCondition,
            /*hasElse=*/true);

        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
          builder.create<scf::YieldOp>(loc, pattern.baseInputs[baseIdx]);
        }

        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
          builder.create<scf::YieldOp>(loc, result);
        }

        result = ifOp.getResult(0);
      }
    }

    return result;
  }

  // Create conditional mul_plain and add operations based on exceptions
  Value createConditionalMulPlainAndAdd(OpBuilder& builder, Location loc,
                                        Value inductionVar, Value rotResult,
                                        Value currentAcc,
                                        const LinearTransformPattern& pattern) {
    // Build exception set for quick lookup
    llvm::DenseSet<int> skipMulPlainIndices;
    llvm::DenseSet<int> skipAddIndices;

    for (const auto& exc : pattern.exceptions) {
      if (exc.skipMulPlain) skipMulPlainIndices.insert(exc.loopIndex);
      if (exc.skipAdd) skipAddIndices.insert(exc.loopIndex);
    }

    Value currentResult = rotResult;

    // Handle mul_plain with exceptions
    if (!pattern.mulPlainOps.empty() && !skipMulPlainIndices.empty()) {
      // Create conditional mul_plain
      Value normalCase =
          createNormalMulPlain(builder, loc, inductionVar, rotResult, pattern);

      // Build condition for skipping mul_plain
      Value skipCondition;
      bool first = true;
      for (int skipIdx : skipMulPlainIndices) {
        Value skipConstant =
            builder.create<arith::ConstantIndexOp>(loc, skipIdx);
        Value iterCondition = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, inductionVar, skipConstant);

        if (first) {
          skipCondition = iterCondition;
          first = false;
        } else {
          skipCondition =
              builder.create<arith::OrIOp>(loc, skipCondition, iterCondition);
        }
      }

      auto ifOp = builder.create<scf::IfOp>(loc, normalCase.getType(),
                                            skipCondition, /*hasElse=*/true);

      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        builder.create<scf::YieldOp>(loc, rotResult);  // Skip mul_plain
      }

      {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        builder.create<scf::YieldOp>(loc, normalCase);  // Normal mul_plain
      }

      currentResult = ifOp.getResult(0);
    } else if (!pattern.mulPlainOps.empty()) {
      currentResult =
          createNormalMulPlain(builder, loc, inductionVar, rotResult, pattern);
    }

    // Handle add with exceptions (similar pattern)
    if (currentAcc && !pattern.addOps.empty()) {
      if (!skipAddIndices.empty()) {
        // Create conditional add
        Value addResult = builder.create<openfhe::AddOp>(
            loc, currentAcc.getType(),
            const_cast<openfhe::AddOp&>(pattern.addOps[0]).getCryptoContext(),
            currentAcc, currentResult);

        // Build condition for skipping add
        Value skipCondition;
        bool first = true;
        for (int skipIdx : skipAddIndices) {
          Value skipConstant =
              builder.create<arith::ConstantIndexOp>(loc, skipIdx);
          Value iterCondition = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, inductionVar, skipConstant);

          if (first) {
            skipCondition = iterCondition;
            first = false;
          } else {
            skipCondition =
                builder.create<arith::OrIOp>(loc, skipCondition, iterCondition);
          }
        }

        auto ifOp = builder.create<scf::IfOp>(loc, currentAcc.getType(),
                                              skipCondition, /*hasElse=*/true);

        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
          builder.create<scf::YieldOp>(loc, currentAcc);  // Skip add
        }

        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
          builder.create<scf::YieldOp>(loc, addResult);  // Normal add
        }

        return ifOp.getResult(0);
      } else {
        return builder.create<openfhe::AddOp>(
            loc, currentAcc.getType(),
            const_cast<openfhe::AddOp&>(pattern.addOps[0]).getCryptoContext(),
            currentAcc, currentResult);
      }
    }

    return currentResult;
  }

  Value createNormalMulPlain(OpBuilder& builder, Location loc,
                             Value inductionVar, Value rotResult,
                             const LinearTransformPattern& pattern) {
    // Create plaintext selector
    Value selectedPt =
        createPlaintextSelector(builder, loc, inductionVar, pattern.plaintexts);

    return builder.create<openfhe::MulPlainOp>(
        loc,
        const_cast<openfhe::MulPlainOp&>(pattern.mulPlainOps[0])
            .getResult()
            .getType(),
        const_cast<openfhe::MulPlainOp&>(pattern.mulPlainOps[0])
            .getCryptoContext(),
        rotResult, selectedPt);
  }

  Value createPlaintextSelector(OpBuilder& builder, Location loc,
                                Value inductionVar,
                                ArrayRef<Value> plaintexts) {
    if (plaintexts.size() == 1) {
      return plaintexts[0];
    }

    // Create a simple selection for small lists
    if (plaintexts.size() <= 8) {
      Value result = plaintexts[0];
      for (int i = 1; i < plaintexts.size(); ++i) {
        if (!plaintexts[i]) continue;  // Skip invalid plaintexts

        Value expectedIndex = builder.create<arith::ConstantIndexOp>(loc, i);
        Value condition = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, inductionVar, expectedIndex);

        auto ifOp = builder.create<scf::IfOp>(loc, plaintexts[i].getType(),
                                              condition, /*hasElse=*/true);

        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
          builder.create<scf::YieldOp>(loc, plaintexts[i]);
        }

        {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
          builder.create<scf::YieldOp>(loc, result);
        }

        result = ifOp.getResult(0);
      }
      return result;
    }

    // For large lists, just use the first one for now
    LLVM_DEBUG(llvm::dbgs() << "Too many plaintexts (" << plaintexts.size()
                            << "), using first one only\n");
    return plaintexts[0];
  }

  // Check if a pattern is still valid (operations haven't been erased)
  bool isPatternStillValid(const LinearTransformPattern& pattern) {
    // Check if any of the core operations have been erased
    for (auto deserOp : pattern.deserializeOps) {
      Operation* op =
          const_cast<openfhe::DeserializeKeyOp&>(deserOp).getOperation();
      if (!op || !op->getParentOp()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Pattern invalid: deserialize operation erased\n");
        return false;
      }
    }

    for (auto rotOp : pattern.rotOps) {
      Operation* op = const_cast<openfhe::RotOp&>(rotOp).getOperation();
      if (!op || !op->getParentOp()) {
        LLVM_DEBUG(llvm::dbgs() << "Pattern invalid: rot operation erased\n");
        return false;
      }
    }

    return true;
  }

  // Invalidate patterns that might conflict with a recently transformed pattern
  void invalidateConflictingPatterns(
      SmallVector<LinearTransformPattern, 4>& patterns, int startIndex,
      const LinearTransformPattern& transformedPattern) {
    llvm::DenseSet<Operation*> transformedOps;

    // Collect operations from the transformed pattern
    for (auto deserOp : transformedPattern.deserializeOps) {
      transformedOps.insert(
          const_cast<openfhe::DeserializeKeyOp&>(deserOp).getOperation());
    }
    for (auto rotOp : transformedPattern.rotOps) {
      transformedOps.insert(const_cast<openfhe::RotOp&>(rotOp).getOperation());
    }

    // Mark conflicting patterns as invalid by clearing their operations
    for (int i = startIndex; i < patterns.size(); ++i) {
      auto& pattern = patterns[i];
      bool hasConflict = false;

      // Check for conflicts with deserialize operations
      for (auto deserOp : pattern.deserializeOps) {
        if (transformedOps.contains(
                const_cast<openfhe::DeserializeKeyOp&>(deserOp)
                    .getOperation())) {
          hasConflict = true;
          break;
        }
      }

      // Check for conflicts with rot operations
      if (!hasConflict) {
        for (auto rotOp : pattern.rotOps) {
          if (transformedOps.contains(
                  const_cast<openfhe::RotOp&>(rotOp).getOperation())) {
            hasConflict = true;
            break;
          }
        }
      }

      if (hasConflict) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Invalidating conflicting pattern " << i << "\n");
        // Clear the pattern to make it invalid
        pattern.deserializeOps.clear();
        pattern.rotOps.clear();
      }
    }
  }

  void cleanupLinearTransformPattern(const LinearTransformPattern& pattern) {
    LLVM_DEBUG(llvm::dbgs()
               << "Cleaning up pattern with " << pattern.deserializeOps.size()
               << " iterations\n");

    // Create a work list of operations to potentially erase
    SmallVector<Operation*, 32> candidatesForErasure;

    // Collect enqueue operations (safe to erase since they were hoisted)
    for (auto enqueueOp : pattern.enqueueOps) {
      Operation* op = enqueueOp.getOperation();
      if (op && op->getParentOp() && op->use_empty()) {
        candidatesForErasure.push_back(op);
      }
    }

    // Collect clear operations (usually safe to erase)
    for (Operation* clearOp : pattern.clearOps) {
      if (clearOp && clearOp->getParentOp() && clearOp->use_empty()) {
        candidatesForErasure.push_back(clearOp);
      }
    }

    // For computational operations, be more careful
    // Only erase if they have no uses and are not the deserialize operations

    // Add operations (highest level in dependency chain)
    for (auto addOp : pattern.addOps) {
      if (addOp) {
        Operation* op = const_cast<openfhe::AddOp&>(addOp).getOperation();
        if (op && op->getParentOp() && op->use_empty()) {
          candidatesForErasure.push_back(op);
        } else if (op && !op->use_empty()) {
          LLVM_DEBUG(llvm::dbgs()
                     << "Skipping add operation with uses: " << *op << "\n");
        }
      }
    }

    // Mul plain operations
    for (auto mulOp : pattern.mulPlainOps) {
      if (mulOp) {
        Operation* op = const_cast<openfhe::MulPlainOp&>(mulOp).getOperation();
        if (op && op->getParentOp() && op->use_empty()) {
          candidatesForErasure.push_back(op);
        } else if (op && !op->use_empty()) {
          LLVM_DEBUG(llvm::dbgs() << "Skipping mul_plain operation with uses: "
                                  << *op << "\n");
        }
      }
    }

    // Rot operations
    for (auto rotOp : pattern.rotOps) {
      Operation* op = const_cast<openfhe::RotOp&>(rotOp).getOperation();
      if (op && op->getParentOp() && op->use_empty()) {
        candidatesForErasure.push_back(op);
      } else if (op && !op->use_empty()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Skipping rot operation with uses: " << *op << "\n");
      }
    }

    // Note: Explicitly NOT erasing deserialize operations per your requirement

    // Now safely erase the collected operations
    for (Operation* op : candidatesForErasure) {
      if (op && op->getParentOp() && op->use_empty()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Safely erasing operation: " << op->getName() << "\n");
        op->erase();
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "Cleanup complete, erased "
                            << candidatesForErasure.size() << " operations\n");
  }
};

}  // namespace heir
}  // namespace mlir
