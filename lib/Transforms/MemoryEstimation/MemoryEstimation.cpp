#include "MemoryEstimation.h"

#include <algorithm>
#include <iomanip>
#include <map>
#include <sstream>

#include "lib/Analysis/UnifiedLivenessAnalysis/UnifiedLivenessAnalysis.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
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

// Forward declaration for clean separation
struct OperationLivenessInfo;
class UnifiedLivenessResults;

// Get Chebyshev degree from operation attributes
static unsigned getChebyshevDegree(openfhe::ChebyshevOp op) {
  // Try to get degree from coefficients attribute
  if (auto coeffsAttr = op->getAttrOfType<ArrayAttr>("coefficients")) {
    return coeffsAttr.size();
  }
  return 32;  // Default medium degree if not found
}

// Calculate memory size for keys using formula: 2×64Ndnum(k+k′)/8=16Ndnum(k+k′)
constexpr static uint64_t calculateKeyMemoryBytes(unsigned qTowers,
                                                  unsigned pTowers,
                                                  unsigned ringDim,
                                                  unsigned dnum = 2) {
  return 16ULL * ringDim * dnum * (qTowers + pTowers);
}

// Calculate memory size for ciphertexts using formula: 2×64Ndk/8=16Ndk
constexpr static uint64_t calculateCiphertextMemoryBytes(unsigned qTowers,
                                                         unsigned ringDim,
                                                         unsigned dnum = 2) {
  return 16ULL * ringDim * dnum * qTowers;
}

// Memory usage information for a single operation
struct MemoryUsage {
  uint64_t totalBytes = 0;
  uint64_t ciphertextBytes = 0;
  uint64_t keyBytes = 0;
  unsigned liveCiphertexts = 0;
  unsigned liveKeys = 0;
  unsigned totalTowers = 0;
};

// Results from memory estimation analysis
struct MemoryEstimationResults {
  // Summary statistics
  uint64_t peakMemoryBytes = 0;
  uint64_t averageMemoryBytes = 0;
  uint64_t totalCiphertextMemory = 0;
  uint64_t totalKeyMemory = 0;

  // Parameters used for calculation
  unsigned ringDimension = 4096;
  unsigned dnumParameter = 2;

  // Timestep data for matrix visualization
  std::map<unsigned, unsigned> timestepTowerCounts;  // timestep -> total_towers
  std::map<unsigned, unsigned>
      timestepCiphertextTowers;  // timestep -> ciphertext_towers
  std::map<unsigned, unsigned> timestepKeyTowers;  // timestep -> key_towers
  std::map<unsigned, double>
      timestepScaledTime;  // timestep -> scaled_time (optional)

  // Report generation methods
  void generateTextReport(llvm::raw_ostream &os) const;
  void generateMatrixReport(llvm::raw_ostream &os) const;
  void generateJSONReport(llvm::raw_ostream &os) const;

  std::string formatBytes(uint64_t bytes) const;
};

//===----------------------------------------------------------------------===//
// Operation Cost Helpers (Improved with realistic modeling)
//===----------------------------------------------------------------------===//

// Chebyshev depth lookup table based on OpenFHE analysis
static const std::map<unsigned, unsigned> CHEBYSHEV_DEPTH_TABLE = {
    {5, 4},    {6, 5},    {13, 5},    {14, 6},    {27, 6},   {28, 7},
    {59, 7},   {60, 8},   {119, 8},   {120, 9},   {247, 9},  {248, 10},
    {495, 10}, {496, 11}, {1007, 11}, {1008, 12}, {2031, 12}};

static unsigned getDepthFromTable(unsigned degree) {
  for (const auto &[maxDegree, depth] : CHEBYSHEV_DEPTH_TABLE) {
    if (degree <= maxDegree) return depth;
  }
  return 12;  // Default for very high degrees
}

// Memory pressure scaling based on tower count
static double calculateTowerMemoryMultiplier(unsigned towers) {
  // Memory pressure increases non-linearly with tower count
  // Based on actual OpenFHE memory patterns
  if (towers <= 5) return 1.0;
  if (towers <= 10) return 1.2;
  if (towers <= 20) return 1.5;
  if (towers <= 30) return 2.0;
  return 2.5;  // Very high tower counts
}

// Improved operation cost table with realistic ratios
static int64_t getOperationCost(Operation *op) {
  return llvm::TypeSwitch<Operation *, int64_t>(op)
      // Basic operations (baseline)
      .Case<openfhe::AddOp, openfhe::SubOp, openfhe::AddPlainOp,
            openfhe::SubPlainOp>([](auto) { return 1; })

      // Multiplication operations (more realistic ratios)
      .Case<openfhe::MulPlainOp>(
          [](auto) { return 3; })  // Just polynomial mult
      .Case<openfhe::MulNoRelinOp>(
          [](auto) { return 5; })                     // Mult without keyswitch
      .Case<openfhe::MulOp>([](auto) { return 18; })  // Mult + relinearization

      // Key management operations
      .Case<openfhe::RelinOp>(
          [](auto) { return 12; })  // Standalone relinearization
      .Case<openfhe::KeySwitchOp>([](auto) { return 15; })  // Key switching
      .Case<openfhe::RotOp, openfhe::AutomorphOp>(
          [](auto) { return 20; })  // Rotation + keyswitch

      // Level management
      .Case<openfhe::ModReduceOp>([](auto) { return 2; })    // Simple rescaling
      .Case<openfhe::LevelReduceOp>([](auto) { return 1; })  // Tower dropping

      // Chebyshev with dynamic cost based on degree
      .Case<openfhe::ChebyshevOp>([](auto op) {
        unsigned degree = getChebyshevDegree(op);
        if (degree < 5) return static_cast<int64_t>(15);  // Linear method

        // PS method cost calculation based on actual operation count
        unsigned k = static_cast<unsigned>(std::ceil(std::sqrt(degree / 2.0)));
        unsigned m = static_cast<unsigned>(std::ceil(std::log2(degree / k)));
        unsigned mults = k + 2 * m + (1 << (m - 1)) - 4;  // PS formula
        return static_cast<int64_t>(mults * 18 +
                                    50);  // mults * MulOp_cost + overhead
      })

      // Bootstrap operation (much more realistic)
      .Case<openfhe::BootstrapOp>([](auto) { return 25000; })

      // Zero-cost operations
      .Case<openfhe::DeserializeKeyOp, openfhe::SerializeKeyOp,
            openfhe::ClearKeyOp, openfhe::EnqueueKeyOp, openfhe::ClearCtOp,
            openfhe::CompressKeyOp>([](auto) { return 0; })

      .Default([](Operation *) { return 1; });
}

// Get towers from result_towers attribute or default
static unsigned getResultTowers(Operation *op) {
  if (auto towersAttr = op->getAttrOfType<IntegerAttr>("result_towers")) {
    return towersAttr.getInt();
  }
  // Default based on operation type
  if (isa<openfhe::BootstrapOp>(op)) return 18;  // Realistic bootstrap output
  if (isa<openfhe::MulOp>(op)) return 5;         // Multiplication reduces level
  return 3;                                      // Default
}

// Improved operation time calculation with tower-aware scaling
static double calculateOperationTime(Operation *op) {
  int64_t baseCost = getOperationCost(op);

  // For bootstrap, use fixed cost (not tower-dependent)
  if (isa<openfhe::BootstrapOp>(op)) {
    return static_cast<double>(baseCost);
  }

  unsigned towers = getResultTowers(op);

  // Different scaling for different operations
  double towerScale = 1.0;
  if (isa<openfhe::MulOp>(op) || isa<openfhe::RelinOp>(op)) {
    // Multiplication and relinearization scale more with towers
    towerScale = calculateTowerMemoryMultiplier(towers);
  } else if (isa<openfhe::RotOp, openfhe::AutomorphOp>(op)) {
    // Rotations have high fixed cost, less tower dependence
    towerScale = 1.0 + 0.1 * (towers / 5.0);
  } else {
    // Linear operations scale mildly
    towerScale = 1.0 + 0.05 * (towers / 5.0);
  }

  return static_cast<double>(baseCost) * towerScale;
}

// Precise Chebyshev peak memory calculation based on our analysis
static unsigned calculateChebyshevPeakTowers(unsigned degree,
                                             unsigned inputTowers) {
  if (degree <= 1) return inputTowers;
  if (degree < 5) return degree + 2;  // Linear method: k + 2

  // PS method with precise formula
  unsigned k = static_cast<unsigned>(std::ceil(std::sqrt(degree / 2.0)));
  unsigned m = static_cast<unsigned>(std::ceil(std::log2(degree / k)));
  unsigned totalDepth = getDepthFromTable(degree);

  // Precise tower calculation based on our detailed analysis
  unsigned logK = static_cast<unsigned>(std::ceil(std::log2(k)));
  unsigned baseTowerLevel =
      (inputTowers > logK + 2) ? inputTowers - logK - 2 : 1;

  // T vector towers
  unsigned tVectorTowers = k * baseTowerLevel;

  // T2 vector towers (decreasing by 1 each level)
  unsigned t2VectorTowers = 0;
  for (unsigned i = 0; i < m; i++) {
    if (baseTowerLevel > i) {
      t2VectorTowers += (baseTowerLevel - i);
    }
  }

  // Intermediate towers
  unsigned intermediateTowers = 5 * std::max(1u, inputTowers - totalDepth + 2);

  return tVectorTowers + t2VectorTowers + intermediateTowers;
}

// Significantly improved special operation memory modeling
static void adjustSpecialOperationMemory(
    const OperationLivenessInfo &operationInfo, MemoryUsage &usage) {
  Operation *op = operationInfo.op;

  if (auto bootstrapOp = dyn_cast<openfhe::BootstrapOp>(op)) {
    // Bootstrap memory pattern based on detailed analysis
    unsigned inputTowers = getResultTowers(op);

    // Bootstrap extends to near-full depth (~30), then processes
    unsigned extendedTowers = 30;  // After modulus extension

    // Chebyshev (degree 84) peak memory: ~25 ciphertexts with varying towers
    unsigned chebyshevPeakTowers = 25 * 18;  // ~25 ciphertexts * 18 avg towers

    // Main bootstrap variables: 6-8 ciphertexts with high tower counts
    unsigned mainVarTowers = 6 * extendedTowers;

    // Total bootstrap peak memory
    unsigned totalBootstrapTowers = chebyshevPeakTowers + mainVarTowers;

    usage.ciphertextBytes +=
        calculateCiphertextMemoryBytes(totalBootstrapTowers, 4096, 2);
    usage.totalTowers += totalBootstrapTowers;

  } else if (auto chebyshevOp = dyn_cast<openfhe::ChebyshevOp>(op)) {
    // Precise Chebyshev memory calculation
    unsigned degree = getChebyshevDegree(chebyshevOp);
    unsigned inputTowers = getResultTowers(op);
    unsigned peakTowers = calculateChebyshevPeakTowers(degree, inputTowers);

    // Add the peak memory during Chebyshev computation
    usage.ciphertextBytes +=
        calculateCiphertextMemoryBytes(peakTowers, 4096, 2);
    usage.totalTowers += peakTowers;
  }
}

//===----------------------------------------------------------------------===//
// MemoryEstimationResults Implementation
//===----------------------------------------------------------------------===//

void MemoryEstimationResults::generateTextReport(llvm::raw_ostream &os) const {
  os << "=== Memory Estimation Report ===\n\n";

  os << "Summary:\n";
  os << "  Peak Memory: " << formatBytes(peakMemoryBytes) << "\n";
  os << "  Average Memory: " << formatBytes(averageMemoryBytes) << "\n";
  os << "  Total Ciphertext Memory: " << formatBytes(totalCiphertextMemory)
     << "\n";
  os << "  Total Key Memory: " << formatBytes(totalKeyMemory) << "\n";
  os << "  Ring Dimension: " << ringDimension << "\n";
  os << "  Dnum Parameter: " << dnumParameter << "\n\n";

  os << "Note: Detailed per-timestep breakdown is shown in the debug output "
        "above.\n";
  os << "=== End Report ===\n";
}

void MemoryEstimationResults::generateMatrixReport(
    llvm::raw_ostream &os) const {
  os << "=== Memory Visualization Matrix ===\n\n";

  os << "Legend: Each * represents ~100 towers\n";
  os << "Format: Timestep | Total Towers | Visual\n\n";

  for (const auto &[timestep, towers] : timestepTowerCounts) {
    // Calculate number of stars (each star = ~100 towers)
    unsigned stars = (towers + 50) / 100;  // Round to nearest 100

    os << "T";
    if (timestep < 10)
      os << "   ";
    else if (timestep < 100)
      os << "  ";
    else if (timestep < 1000)
      os << " ";
    os << timestep << " | ";

    if (towers < 10)
      os << "     ";
    else if (towers < 100)
      os << "    ";
    else if (towers < 1000)
      os << "   ";
    else if (towers < 10000)
      os << "  ";
    else if (towers < 100000)
      os << " ";
    os << towers << " | ";

    // Print stars
    for (unsigned i = 0; i < stars && i < 80; ++i) {  // Limit to 80 chars width
      os << "*";
    }
    if (stars > 80) {
      os << "... (" << stars << " total)";
    }
    os << "\n";
  }

  os << "\n=== End Matrix ===\n";
}

void MemoryEstimationResults::generateJSONReport(llvm::raw_ostream &os) const {
  os << "{\n";
  os << "  \"summary\": {\n";
  os << "    \"peakMemoryBytes\": " << peakMemoryBytes << ",\n";
  os << "    \"averageMemoryBytes\": " << averageMemoryBytes << ",\n";
  os << "    \"totalCiphertextMemory\": " << totalCiphertextMemory << ",\n";
  os << "    \"totalKeyMemory\": " << totalKeyMemory << ",\n";
  os << "    \"ringDimension\": " << ringDimension << ",\n";
  os << "    \"dnumParameter\": " << dnumParameter << "\n";
  os << "  },\n";

  os << "  \"timesteps\": [\n";
  bool first = true;
  for (const auto &[timestep, towers] : timestepTowerCounts) {
    if (!first) os << ",\n";
    first = false;

    os << "    {\n";
    os << "      \"timestep\": " << timestep << ",\n";
    os << "      \"totalTowers\": " << towers << ",\n";
    os << "      \"ciphertextTowers\": "
       << timestepCiphertextTowers.at(timestep) << ",\n";
    os << "      \"keyTowers\": " << timestepKeyTowers.at(timestep);

    // Add scaled time if available
    auto timeIt = timestepScaledTime.find(timestep);
    if (timeIt != timestepScaledTime.end()) {
      os << ",\n      \"scaledTime\": " << timeIt->second;
    }

    os << "\n    }";
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

    uint64_t totalMemorySum = 0;
    unsigned operationCount = 0;
    double cumulativeTime = 0.0;  // For time scaling

    // Process each operation's liveness information IN TIMESTEP ORDER
    for (const auto &operationInfo : livenessResults.operationResults) {
      operationCount++;

      // Extract liveness data using the clear API
      MemoryUsage usage;
      usage.liveCiphertexts = operationInfo.liveCiphertextCount;
      usage.liveKeys = operationInfo.liveKeyCount;
      usage.totalTowers = operationInfo.totalTowers;

      // **CRITICAL: Add validation for rotation operations**
      if (isa<openfhe::RotOp>(operationInfo.op)) {
        if (usage.liveKeys == 0) {
          operationInfo.op->emitError(
              "Liveness analysis shows rotation operation with ")
              << usage.liveKeys << " live keys";
          return signalPassFailure();
        }
      }

      // Calculate memory usage based on EXACT tower information from liveness
      // Ciphertext memory: use exact tower counts from liveness analysis
      usage.ciphertextBytes = 0;
      for (unsigned towers : operationInfo.ciphertextTowerCounts) {
        usage.ciphertextBytes +=
            calculateCiphertextMemoryBytes(towers, ringDim, dnum);
      }

      // Key memory: use exact tower counts from liveness analysis
      usage.keyBytes = 0;
      for (unsigned towers : operationInfo.keyTowerCounts) {
        // For keys, the tower count from liveness already includes Q+P towers
        // Split into Q and P for memory calculation (assume fixed P towers)
        unsigned qTowers =
            (towers > fixedPTowers) ? towers - fixedPTowers : towers;
        unsigned pTowers = (towers > fixedPTowers) ? fixedPTowers : 0;
        usage.keyBytes +=
            calculateKeyMemoryBytes(qTowers, pTowers, ringDim, dnum);
      }

      usage.totalBytes = usage.ciphertextBytes + usage.keyBytes;

      // **IMPROVED: Add special memory adjustments for complex operations**
      adjustSpecialOperationMemory(operationInfo, usage);

      // Track statistics
      totalMemorySum += usage.totalBytes;
      if (usage.totalBytes > results.peakMemoryBytes) {
        results.peakMemoryBytes = usage.totalBytes;
      }
      results.totalCiphertextMemory += usage.ciphertextBytes;
      results.totalKeyMemory += usage.keyBytes;

      // Calculate operation time with improved scaling (if opScale is enabled)
      if (opScale) {
        double opTime = calculateOperationTime(operationInfo.op);

        // **NEW: Apply memory pressure scaling**
        double memoryPressure = static_cast<double>(usage.totalTowers) / 1000.0;
        if (memoryPressure > 2.0) {  // High memory pressure
          // Operations take longer under memory pressure
          opTime *= (1.0 + 0.2 * (memoryPressure - 2.0));
        }

        cumulativeTime += opTime;
        results.timestepScaledTime[operationInfo.timestep] = cumulativeTime;
      }

      // Store timestep data for matrix visualization
      // Use the adjusted tower counts that include special operation memory
      results.timestepTowerCounts[operationInfo.timestep] = usage.totalTowers;
      results.timestepCiphertextTowers[operationInfo.timestep] =
          operationInfo.totalCiphertextTowers;
      results.timestepKeyTowers[operationInfo.timestep] =
          operationInfo.totalKeyTowers;

      // For bootstrap and Chebyshev, add their special tower contributions to
      // the visualization
      if (isa<openfhe::BootstrapOp>(operationInfo.op)) {
        results.timestepCiphertextTowers[operationInfo.timestep] +=
            750;  // More realistic bootstrap towers
      } else if (auto chebyshevOp =
                     dyn_cast<openfhe::ChebyshevOp>(operationInfo.op)) {
        unsigned degree = getChebyshevDegree(chebyshevOp);
        unsigned inputTowers = getResultTowers(operationInfo.op);
        unsigned peakTowers = calculateChebyshevPeakTowers(degree, inputTowers);
        results.timestepCiphertextTowers[operationInfo.timestep] += peakTowers;
      }
    }

    if (operationCount > 0) {
      results.averageMemoryBytes = totalMemorySum / operationCount;
    }

    // Generate output based on format
    if (outputFormat == "matrix") {
      if (!outputFile.empty()) {
        std::error_code EC;
        llvm::raw_fd_ostream file(outputFile, EC);
        if (!EC) {
          results.generateMatrixReport(file);
        } else {
          results.generateMatrixReport(llvm::errs());
        }
      } else {
        results.generateMatrixReport(llvm::errs());
      }
    } else if (outputFormat == "json") {
      if (!outputFile.empty()) {
        std::error_code EC;
        llvm::raw_fd_ostream file(outputFile, EC);
        if (!EC) {
          results.generateJSONReport(file);
        } else {
          results.generateJSONReport(llvm::errs());
        }
      } else {
        results.generateJSONReport(llvm::errs());
      }
    } else {
      // Default text output
      if (!outputFile.empty()) {
        std::error_code EC;
        llvm::raw_fd_ostream file(outputFile, EC);
        if (!EC) {
          results.generateTextReport(file);
        } else {
          results.generateTextReport(llvm::errs());
        }
      } else {
        results.generateTextReport(llvm::errs());
      }
    }
  }
};

}  // namespace

}  // namespace heir
}  // namespace mlir
