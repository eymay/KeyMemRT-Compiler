#include "MemoryEstimation.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

#include "lib/Analysis/UnifiedLivenessAnalysis/UnifiedLivenessAnalysis.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "mlir/include/mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"               // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"       // from @llvm-project

#define DEBUG_TYPE "memory-estimation"

namespace mlir {
namespace heir {

//===----------------------------------------------------------------------===//
// Memory Calculation Helpers
//===----------------------------------------------------------------------===//

// Calculate memory size for keys using formula: 2×64Ndnum(k+k′)/8=16Ndnum(k+k′)
static uint64_t calculateKeyMemoryBytes(unsigned qTowers, unsigned pTowers,
                                        unsigned ringDim, unsigned dnum = 2) {
  return 16ULL * ringDim * dnum * (qTowers + pTowers);
}

// Calculate memory size for ciphertexts using formula: 2×64Ndk/8=16Ndk
static uint64_t calculateCiphertextMemoryBytes(unsigned qTowers,
                                               unsigned ringDim,
                                               unsigned dnum = 2) {
  return 16ULL * ringDim * dnum * qTowers;
}

//===----------------------------------------------------------------------===//
// MemoryEstimationResults Implementation
//===----------------------------------------------------------------------===//

void MemoryEstimationResults::generateTextReport(llvm::raw_ostream &os) const {
  os << "=== Memory Estimation Report ===\n\n";

  os << "Summary:\n";
  os << "  Total operations analyzed: " << operationMemoryUsage.size() << "\n";
  os << "  Peak memory usage: " << formatBytes(peakMemoryBytes) << "\n";
  os << "  Average memory usage: " << formatBytes(averageMemoryBytes) << "\n";
  os << "  Ring dimension (N): " << ringDimension << "\n";
  os << "  Dnum parameter: " << dnumParameter << "\n\n";

  os << "Detailed Operation Analysis:\n";
  unsigned opIndex = 0;
  for (const auto &[op, usage] : operationMemoryUsage) {
    os << "  Op #" << opIndex++ << " (" << op->getName() << "):\n";
    os << "    Live ciphertexts: " << usage.liveCiphertexts << "\n";
    os << "    Live keys: " << usage.liveKeys << "\n";
    os << "    Total memory: " << formatBytes(usage.totalBytes) << "\n";
    os << "    Ciphertext memory: " << formatBytes(usage.ciphertextBytes)
       << "\n";
    os << "    Key memory: " << formatBytes(usage.keyBytes) << "\n\n";
  }

  os << "=== End Report ===\n";
}

void MemoryEstimationResults::generateJSONReport(llvm::raw_ostream &os) const {
  os << "{\n";
  os << "  \"analysis_type\": \"memory_estimation\",\n";
  os << "  \"parameters\": {\n";
  os << "    \"ring_dimension\": " << ringDimension << ",\n";
  os << "    \"dnum_parameter\": " << dnumParameter << "\n";
  os << "  },\n";
  os << "  \"summary\": {\n";
  os << "    \"total_operations\": " << operationMemoryUsage.size() << ",\n";
  os << "    \"peak_memory_bytes\": " << peakMemoryBytes << ",\n";
  os << "    \"average_memory_bytes\": " << averageMemoryBytes << "\n";
  os << "  },\n";

  os << "  \"operations\": [\n";
  bool first = true;
  unsigned opIndex = 0;
  for (const auto &[op, usage] : operationMemoryUsage) {
    if (!first) os << ",\n";
    first = false;

    os << "    {\n";
    os << "      \"index\": " << opIndex++ << ",\n";
    os << "      \"operation\": \"" << op->getName().getStringRef() << "\",\n";
    os << "      \"total_memory_bytes\": " << usage.totalBytes << ",\n";
    os << "      \"ciphertext_memory_bytes\": " << usage.ciphertextBytes
       << ",\n";
    os << "      \"key_memory_bytes\": " << usage.keyBytes << ",\n";
    os << "      \"live_ciphertexts\": " << usage.liveCiphertexts << ",\n";
    os << "      \"live_keys\": " << usage.liveKeys << ",\n";
    os << "      \"total_towers\": " << usage.totalTowers << "\n";
    os << "    }";
  }
  os << "\n  ]\n";
  os << "}\n";
}

std::string MemoryEstimationResults::formatBytes(uint64_t bytes) const {
  const char *units[] = {"B", "KB", "MB", "GB", "TB"};
  int unitIndex = 0;
  double size = static_cast<double>(bytes);

  while (size >= 1024.0 && unitIndex < 4) {
    size /= 1024.0;
    unitIndex++;
  }

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << size << " " << units[unitIndex];
  return oss.str();
}

//===----------------------------------------------------------------------===//
// MemoryEstimation Pass Implementation
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_MEMORYESTIMATION
#include "lib/Transforms/MemoryEstimation/MemoryEstimation.h.inc"

namespace {

struct MemoryEstimationPass : impl::MemoryEstimationBase<MemoryEstimationPass> {
  using MemoryEstimationBase::MemoryEstimationBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    // Run the unified liveness analysis and get structured results
    UnifiedLivenessResults livenessResults = runUnifiedLivenessAnalysis(op);

    // Collect memory estimation results
    MemoryEstimationResults results;
    results.ringDimension = ringDim;
    results.dnumParameter = dnum;
    results.showMatrix = (outputFormat == "matrix");

    uint64_t totalMemory = 0;
    unsigned operationCount = 0;

    // Process each operation's liveness information
    for (const auto &livenessInfo : livenessResults.operationResults) {
      MemoryUsage usage;
      usage.liveCiphertexts = livenessInfo.liveCiphertextCount;
      usage.liveKeys = livenessInfo.liveKeyCount;
      usage.totalTowers = livenessInfo.totalTowers;

      // Calculate ciphertext memory
      for (unsigned towers : livenessInfo.ciphertextTowerCounts) {
        uint64_t ctMemory =
            calculateCiphertextMemoryBytes(towers, ringDim, dnum);
        usage.ciphertextBytes += ctMemory;
      }

      // Calculate key memory using rotation indices and depths
      for (size_t i = 0; i < livenessInfo.keyTowerCounts.size(); ++i) {
        unsigned towers = livenessInfo.keyTowerCounts[i];

        // Extract Q and P towers from total towers
        unsigned qTowers = (towers > fixedPTowers) ? towers - fixedPTowers : 1;
        unsigned pTowers = fixedPTowers;

        // If we have key depth information, use it for more accurate
        // calculation
        if (i < livenessInfo.keyDepths.size()) {
          unsigned keyDepth = livenessInfo.keyDepths[i];
          qTowers = calculateQTowersFromKeyDepth(keyDepth);
        }

        uint64_t keyMemory =
            calculateKeyMemoryBytes(qTowers, pTowers, ringDim, dnum);
        usage.keyBytes += keyMemory;
      }

      usage.totalBytes = usage.ciphertextBytes + usage.keyBytes;
      results.operationMemoryUsage[livenessInfo.op] = usage;

      // Update statistics
      totalMemory += usage.totalBytes;
      operationCount++;
      results.peakMemoryBytes =
          std::max(results.peakMemoryBytes, usage.totalBytes);
    }

    // Calculate averages
    if (operationCount > 0) {
      results.averageMemoryBytes = totalMemory / operationCount;
    }

    // Generate output based on format
    if (outputFormat == "json") {
      if (outputFile.empty()) {
        results.generateJSONReport(llvm::errs());
      } else {
        std::error_code EC;
        llvm::raw_fd_ostream file(outputFile, EC);
        if (EC) {
          op->emitError("Failed to open output file: " + outputFile);
          signalPassFailure();
          return;
        }
        results.generateJSONReport(file);
      }
    } else {
      if (outputFile.empty()) {
        results.generateTextReport(llvm::errs());
      } else {
        std::error_code EC;
        llvm::raw_fd_ostream file(outputFile, EC);
        if (EC) {
          op->emitError("Failed to open output file: " + outputFile);
          signalPassFailure();
          return;
        }
        results.generateTextReport(file);
      }
    }
  }

 private:
  unsigned calculateQTowersFromKeyDepth(unsigned keyDepth) {
    // Formula: mult_depth - level + 2
    if (keyDepth > multDepth + 2) {
      return 1;  // Minimum towers
    }
    return multDepth - keyDepth + 2;
  }
};

}  // namespace

}  // namespace heir
}  // namespace mlir
