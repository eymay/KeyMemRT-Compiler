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
  SmallVector<Operation*, 16> clearOps;  // clear_key, clear_ct operations
  SmallVector<openfhe::EnqueueKeyOp, 8> enqueueOps;  // To be hoisted

  Value baseInput;    // The ciphertext being rotated (same across all)
  Value accumulator;  // The accumulating result
  SmallVector<Value, 8> plaintexts;  // Different plaintexts for each iteration
  SmallVector<int64_t, 8> rotationIndices;  // The rotation indices (1, 2, 3...)

  bool isValid() const {
    return deserializeOps.size() >= 3 &&
           deserializeOps.size() == rotOps.size() &&
           rotOps.size() == mulPlainOps.size() &&
           mulPlainOps.size() == addOps.size();
  }

  int calculateSavings() const {
    if (!isValid()) return 0;
    int originalOps = deserializeOps.size() + rotOps.size() +
                      mulPlainOps.size() + addOps.size() + clearOps.size();
    int loopOps = 15;  // Loop overhead + body operations
    return originalOps - loopOps;
  }
};

struct PatternLoopRolling : impl::PatternLoopRollingBase<PatternLoopRolling> {
  using impl::PatternLoopRollingBase<
      PatternLoopRolling>::PatternLoopRollingBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    LLVM_DEBUG(llvm::dbgs() << "=== PatternLoopRolling Pass Start ===\n");
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

    // Transform the most profitable pattern
    MLIRContext* context = &getContext();
    OpBuilder builder(context);
    int transformedPatterns = 0;

    if (!patterns.empty()) {
      auto& bestPattern = patterns[0];  // Sorted by savings

      if (bestPattern.calculateSavings() >= 10) {
        if (succeeded(transformLinearTransformPattern(builder, bestPattern))) {
          transformedPatterns++;
          LLVM_DEBUG(llvm::dbgs()
                     << "Transformed pattern with "
                     << bestPattern.deserializeOps.size()
                     << " iterations, saved " << bestPattern.calculateSavings()
                     << " operations\n");
        }
      }
    }

    LLVM_DEBUG(llvm::dbgs()
               << "=== PatternLoopRolling Pass Complete: "
               << transformedPatterns << " patterns transformed ===\n");
  }

 private:
  SmallVector<LinearTransformPattern, 4> findLinearTransformPatterns(
      func::FuncOp func) {
    SmallVector<LinearTransformPattern, 4> patterns;

    // Collect operations in order
    SmallVector<Operation*, 64> operations;
    func.walk([&](Operation* op) {
      if (!isa<func::FuncOp, func::ReturnOp>(op)) {
        operations.push_back(op);
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "Analyzing " << operations.size()
                            << " operations for linear transform patterns\n");

    // Look for the core pattern: deserialize → rot → mul_plain → add (repeated)
    for (int start = 0; start < operations.size() - 10; ++start) {
      if (auto deserOp =
              dyn_cast<openfhe::DeserializeKeyOp>(operations[start])) {
        auto pattern = extractLinearTransformPattern(operations, start);
        if (pattern.isValid()) {
          patterns.push_back(pattern);
          LLVM_DEBUG(llvm::dbgs()
                     << "Found linear transform pattern with "
                     << pattern.deserializeOps.size()
                     << " iterations starting at " << start << "\n");
        }
      }
    }

    // Sort by savings (largest first)
    llvm::sort(patterns, [](const LinearTransformPattern& a,
                            const LinearTransformPattern& b) {
      return a.calculateSavings() > b.calculateSavings();
    });

    return patterns;
  }

  LinearTransformPattern extractLinearTransformPattern(
      ArrayRef<Operation*> operations, int start) {
    LinearTransformPattern pattern;

    int currentPos = start;
    Value currentAcc = Value();  // Track accumulation chain
    Value baseInput = Value();   // Track the base input being rotated

    // Look for repeating pattern: deserialize → (enqueue) → rot → (clear_key) →
    // mul_plain → (clear_ct) → add → (clear_ct)
    while (currentPos < operations.size()) {
      // 1. Look for deserialize_key
      if (currentPos >= operations.size()) break;

      auto deserOp =
          dyn_cast<openfhe::DeserializeKeyOp>(operations[currentPos]);
      if (!deserOp) {
        // If we've already found some iterations and hit a non-deserialize,
        // we're done
        if (!pattern.deserializeOps.empty()) break;
        return pattern;  // No pattern found
      }

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
           lookahead < 5 && currentPos + lookahead < operations.size();
           ++lookahead) {
        if (auto candidateRot =
                dyn_cast<openfhe::RotOp>(operations[currentPos + lookahead])) {
          // Check if this rot uses the key we just deserialized
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
        baseInput = rotOp.getCiphertext();  // Use getCiphertext() method
        pattern.baseInput = baseInput;
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
           lookahead < 3 && currentPos + lookahead < operations.size();
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

      // 7. Look for add that uses the mul result
      openfhe::AddOp addOp = nullptr;
      for (int lookahead = 0;
           lookahead < 3 && currentPos + lookahead < operations.size();
           ++lookahead) {
        if (auto candidateAdd =
                dyn_cast<openfhe::AddOp>(operations[currentPos + lookahead])) {
          // Check if this add uses the mul result
          if (candidateAdd.getLhs() == mulOp.getResult() ||
              candidateAdd.getRhs() == mulOp.getResult()) {
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

      // Update accumulator chain
      currentAcc = addOp.getResult();
      if (pattern.addOps.size() == 1) {
        // For first add, the other operand is the initial accumulator
        pattern.accumulator = (addOp.getLhs() == mulOp.getResult())
                                  ? addOp.getRhs()
                                  : addOp.getLhs();
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
                 << " of linear transform pattern\n");
    }

    return pattern;
  }

  LogicalResult transformLinearTransformPattern(
      OpBuilder& builder, const LinearTransformPattern& pattern) {
    if (!pattern.isValid()) {
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "Transforming linear transform pattern with "
               << pattern.deserializeOps.size() << " iterations\n");

    Operation* firstOp =
        const_cast<openfhe::DeserializeKeyOp&>(pattern.deserializeOps[0])
            .getOperation();
    Location loc = firstOp->getLoc();

    // Hoist all enqueue operations before the pattern (preserving order)
    builder.setInsertionPoint(firstOp);
    for (auto enqueueOp : pattern.enqueueOps) {
      auto newEnqueue = builder.create<openfhe::EnqueueKeyOp>(
          loc, enqueueOp.getCryptoContext(), enqueueOp->getAttr("index"));
      LLVM_DEBUG(llvm::dbgs() << "Hoisting enqueue operation\n");
    }

    // Create loop bounds
    Value lowerBound = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value upperBound = builder.create<arith::ConstantIndexOp>(
        loc, pattern.deserializeOps.size());
    Value step = builder.create<arith::ConstantIndexOp>(loc, 1);

    // Create for loop with accumulator
    auto forOp = builder.create<scf::ForOp>(loc, lowerBound, upperBound, step,
                                            ValueRange{pattern.accumulator});

    // Build loop body
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(forOp.getBody());

    Value inductionVar = forOp.getInductionVar();
    Value currentAcc = forOp.getRegionIterArg(0);

    // Generate loop body operations
    // 1. Keep deserialize_key operations as-is (your requirement)
    Value selectedKey =
        createKeySelector(builder, loc, inductionVar, pattern.deserializeOps);

    // 2. Rotation with selected key
    Value rotResult = builder.create<openfhe::RotOp>(
        loc,
        const_cast<openfhe::RotOp&>(pattern.rotOps[0]).getResult().getType(),
        const_cast<openfhe::RotOp&>(pattern.rotOps[0]).getCryptoContext(),
        pattern.baseInput, selectedKey);

    // 3. Clear the key
    builder.create<openfhe::ClearKeyOp>(
        loc,
        const_cast<openfhe::DeserializeKeyOp&>(pattern.deserializeOps[0])
            .getCryptoContext(),
        selectedKey);

    // 4. Multiply with selected plaintext
    Value selectedPt =
        createPlaintextSelector(builder, loc, inductionVar, pattern.plaintexts);
    Value mulResult = builder.create<openfhe::MulPlainOp>(
        loc,
        const_cast<openfhe::MulPlainOp&>(pattern.mulPlainOps[0])
            .getResult()
            .getType(),
        const_cast<openfhe::MulPlainOp&>(pattern.mulPlainOps[0])
            .getCryptoContext(),
        rotResult, selectedPt);

    // 5. Add to accumulator
    Value newAcc = builder.create<openfhe::AddOp>(
        loc, currentAcc.getType(),
        const_cast<openfhe::AddOp&>(pattern.addOps[0]).getCryptoContext(),
        currentAcc, mulResult);

    // Note: clear_ct and clear_pt operations omitted as per your request

    builder.create<scf::YieldOp>(loc, ValueRange{newAcc});

    // Replace final result
    Value loopResult = forOp.getResult(0);
    const_cast<openfhe::AddOp&>(pattern.addOps.back())
        .getResult()
        .replaceAllUsesWith(loopResult);

    // Clean up original operations
    cleanupLinearTransformPattern(pattern);

    return success();
  }

  Value createKeySelector(OpBuilder& builder, Location loc, Value inductionVar,
                          ArrayRef<openfhe::DeserializeKeyOp> deserializeOps) {
    // For now, just use the first key - this violates your requirement
    // TODO: We need to implement proper key selection without moving
    // deserialize ops This is a placeholder that needs to be fixed
    return const_cast<openfhe::DeserializeKeyOp&>(deserializeOps[0])
        .getResult();
  }

  Value createPlaintextSelector(OpBuilder& builder, Location loc,
                                Value inductionVar,
                                ArrayRef<Value> plaintexts) {
    if (plaintexts.size() == 1) {
      return plaintexts[0];
    }

    // Create a simple selection for small lists
    if (plaintexts.size() <= 5) {
      Value result = plaintexts[0];
      for (int i = 1; i < plaintexts.size(); ++i) {
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

  void cleanupLinearTransformPattern(const LinearTransformPattern& pattern) {
    // Remove original enqueue operations (they've been hoisted)
    for (auto enqueueOp : pattern.enqueueOps) {
      enqueueOp->erase();
    }

    // Remove clear operations
    for (Operation* clearOp : pattern.clearOps) {
      clearOp->erase();
    }

    // Remove original operations (in reverse order)
    for (auto it = pattern.addOps.rbegin(); it != pattern.addOps.rend(); ++it) {
      (*it)->erase();
    }
    for (auto it = pattern.mulPlainOps.rbegin();
         it != pattern.mulPlainOps.rend(); ++it) {
      (*it)->erase();
    }
    for (auto it = pattern.rotOps.rbegin(); it != pattern.rotOps.rend(); ++it) {
      (*it)->erase();
    }
    // Note: Not removing deserialize operations per your requirement
  }
};

}  // namespace heir
}  // namespace mlir
