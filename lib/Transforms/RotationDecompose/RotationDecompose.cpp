#include "lib/Transforms/RotationDecompose/RotationDecompose.h"

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "lib/Analysis/RotationKeyLivenessAnalysis/RotationKeyLivenessAnalysis.h"
#include "lib/Dialect/Openfhe/IR/OpenfheOps.h"
#include "llvm/include/llvm/Support/Debug.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/IRMapping.h"
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"

// Include for getting process ID
#include <unistd.h>

#define DEBUG_TYPE "rotation-decompose"

namespace mlir {
namespace heir {

constexpr bool verbose = false;

namespace {

// Helper to execute shell commands and capture output
std::string exec(const std::string &cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"),
                                                pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return result;
}

// Helper to write data to a temporary file
std::string writeTempFile(const std::string &data,
                          const std::string &suffix = ".json") {
  // Get a unique identifier - either process ID or random number
  unsigned int uniqueId;

#ifdef _WIN32
  // On Windows, use a random number
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned int> distrib(1000, 9999);
  uniqueId = distrib(gen);
#else
  // On POSIX systems, use process ID
  uniqueId = getpid();
#endif

  std::string tmpfile = std::string("/tmp/rotation_optimizer_") +
                        std::to_string(uniqueId) + suffix;
  std::ofstream outfile(tmpfile);
  outfile << data;
  outfile.close();
  return tmpfile;
}

// Structure to hold decomposition information
struct Decomposition {
  int64_t target;
  std::vector<int64_t> baseIndices;
};

// Parse JSON output from Python optimizer
std::pair<std::vector<int64_t>, std::map<int64_t, std::vector<int64_t>>>
parseJsonOutput(const std::string &jsonOutput, int64_t baseSetSize) {
  std::vector<int64_t> baseSet;
  std::map<int64_t, std::vector<int64_t>> decompositions;

  // Basic JSON parsing
  // For base set
  size_t baseSetPos = jsonOutput.find("\"base_set\"");
  if (baseSetPos != std::string::npos) {
    size_t arrayStart = jsonOutput.find('[', baseSetPos);
    size_t arrayEnd = jsonOutput.find(']', arrayStart);

    if (arrayStart != std::string::npos && arrayEnd != std::string::npos) {
      std::string arrayStr =
          jsonOutput.substr(arrayStart + 1, arrayEnd - arrayStart - 1);
      std::stringstream ss(arrayStr);
      std::string item;

      // Parse comma-separated values
      while (std::getline(ss, item, ',')) {
        // Trim whitespace
        item.erase(0, item.find_first_not_of(" \t\n\r"));
        item.erase(item.find_last_not_of(" \t\n\r") + 1);

        if (!item.empty()) {
          baseSet.push_back(std::stoll(item));
        }
      }
    }
  }

  // Parse decompositions
  size_t decompositionsPos = jsonOutput.find("\"decompositions\"");
  if (decompositionsPos != std::string::npos) {
    size_t arrayStart = jsonOutput.find('[', decompositionsPos);
    size_t arrayEnd = jsonOutput.find_last_of(']');

    if (arrayStart != std::string::npos && arrayEnd != std::string::npos) {
      std::string decompositionsStr =
          jsonOutput.substr(arrayStart + 1, arrayEnd - arrayStart - 1);

      // Split the decompositions array into individual objects
      int bracketCount = 0;
      size_t startPos = 0;

      for (size_t i = 0; i < decompositionsStr.length(); ++i) {
        char c = decompositionsStr[i];

        if (c == '{') {
          if (bracketCount == 0) {
            startPos = i;
          }
          bracketCount++;
        } else if (c == '}') {
          bracketCount--;
          if (bracketCount == 0) {
            // Found a complete object
            std::string objStr =
                decompositionsStr.substr(startPos, i - startPos + 1);

            // Parse target
            size_t targetPos = objStr.find("\"target\"");
            if (targetPos != std::string::npos) {
              size_t targetValStart = objStr.find(':', targetPos) + 1;
              size_t targetValEnd = objStr.find(',', targetValStart);
              if (targetValEnd == std::string::npos) {
                targetValEnd = objStr.find('}', targetValStart);
              }

              std::string targetStr =
                  objStr.substr(targetValStart, targetValEnd - targetValStart);
              targetStr.erase(0, targetStr.find_first_not_of(" \t\n\r"));
              targetStr.erase(targetStr.find_last_not_of(" \t\n\r") + 1);

              int64_t target = std::stoll(targetStr);

              // Parse decomposition
              size_t decompPos = objStr.find("\"decomposition\"");
              if (decompPos != std::string::npos) {
                size_t decompArrayStart = objStr.find('[', decompPos);
                size_t decompArrayEnd = objStr.find(']', decompArrayStart);

                if (decompArrayStart != std::string::npos &&
                    decompArrayEnd != std::string::npos) {
                  std::string decompArrayStr =
                      objStr.substr(decompArrayStart + 1,
                                    decompArrayEnd - decompArrayStart - 1);
                  std::stringstream ss(decompArrayStr);
                  std::string item;

                  std::vector<int64_t> baseIndices;

                  // Parse comma-separated values
                  while (std::getline(ss, item, ',')) {
                    // Trim whitespace
                    item.erase(0, item.find_first_not_of(" \t\n\r"));
                    item.erase(item.find_last_not_of(" \t\n\r") + 1);

                    if (!item.empty() && item != "null") {
                      baseIndices.push_back(std::stoll(item));
                    }
                  }

                  decompositions[target] = baseIndices;
                }
              }
            }
          }
        }
      }
    }
  }

  // Validate that we got the expected base set size
  LLVM_DEBUG(if (baseSet.size() != baseSetSize) {
    llvm::dbgs() << "Warning: Expected base set size " << baseSetSize
                 << " but got " << baseSet.size() << "\n";
  });

  return {baseSet, decompositions};
}

// Helper functions for rotation cluster analysis
struct RotationCluster {
  // Operations in the cluster (rotation, deserialize, clear)
  llvm::SmallVector<Operation *, 16> operations;
  // Start and end indices in the block
  unsigned startIdx;
  unsigned endIdx;
  // Distance between operations
  unsigned distance;
  // Set of rotation indices used in this cluster
  llvm::DenseSet<int64_t> rotationIndices;
};
// Analyze rotation clusters based on various heuristics
void analyzeRotationClusters(llvm::ArrayRef<RotationCluster> clusters,
                             const std::vector<int64_t> &baseSet,
                             bool verbose) {
  // Skip if no clusters found
  if (clusters.empty()) {
    if (verbose) {
      llvm::outs() << "No rotation clusters found for analysis\n";
    }
    return;
  }

  if (verbose) {
    llvm::outs() << "Analyzing " << clusters.size() << " rotation clusters\n";
  }

  // Calculate metrics for each cluster
  for (unsigned i = 0; i < clusters.size(); i++) {
    const auto &cluster = clusters[i];

    // Count operations by type
    unsigned rotOpCount = 0;
    unsigned deserOpCount = 0;
    unsigned clearOpCount = 0;
    unsigned otherOpCount = 0;

    for (Operation *op : cluster.operations) {
      if (isa<openfhe::RotOp>(op)) {
        rotOpCount++;
      } else if (isa<openfhe::DeserializeKeyOp>(op)) {
        deserOpCount++;
      } else if (isa<openfhe::ClearKeyOp>(op)) {
        clearOpCount++;
      } else {
        otherOpCount++;
      }
    }

    // Count operations that would need transformation
    unsigned nonBaseIndicesCount = 0;
    for (int64_t idx : cluster.rotationIndices) {
      if (std::find(baseSet.begin(), baseSet.end(), idx) == baseSet.end()) {
        nonBaseIndicesCount++;
      }
    }

    // Calculate an optimization score
    // Higher score = better candidate for optimization
    double optimizationScore = 0.0;

    // More operations = better optimization potential
    optimizationScore += rotOpCount * 2.0;

    // More unique non-base indices = better optimization potential
    optimizationScore += nonBaseIndicesCount * 3.0;

    // Tighter clusters (smaller distances) are better
    if (cluster.distance > 0) {
      optimizationScore += 10.0 / cluster.distance;
    }

    if (verbose) {
      llvm::outs() << "Cluster " << i << " analysis:\n"
                   << "  Operations: " << cluster.operations.size()
                   << " (rotations: " << rotOpCount
                   << ", deserialize: " << deserOpCount
                   << ", clear: " << clearOpCount << ", other: " << otherOpCount
                   << ")\n"
                   << "  Distance: " << cluster.distance << "\n"
                   << "  Rotation indices: ";

      for (auto idx : cluster.rotationIndices) {
        llvm::outs() << idx;
        if (std::find(baseSet.begin(), baseSet.end(), idx) == baseSet.end()) {
          llvm::outs() << "(transform) ";
        } else {
          llvm::outs() << " ";
        }
      }

      llvm::outs() << "\n  Optimization score: " << optimizationScore
                   << (optimizationScore > 10.0
                           ? " (high)"
                           : (optimizationScore > 5.0 ? " (medium)" : " (low)"))
                   << "\n";
    }
  }
}

// Prioritize clusters for optimization
std::vector<unsigned> prioritizeClusters(
    llvm::ArrayRef<RotationCluster> clusters,
    const std::vector<int64_t> &baseSet) {
  // Array of (cluster index, score) pairs
  std::vector<std::pair<unsigned, double>> scoredClusters;

  // Calculate scores for each cluster
  for (unsigned i = 0; i < clusters.size(); i++) {
    const auto &cluster = clusters[i];

    // Skip clusters where all indices are already in the base set
    bool hasNonBaseIndices = false;
    for (int64_t idx : cluster.rotationIndices) {
      if (std::find(baseSet.begin(), baseSet.end(), idx) == baseSet.end()) {
        hasNonBaseIndices = true;
        break;
      }
    }

    if (!hasNonBaseIndices) {
      continue;
    }

    // Count operations by type
    unsigned rotOpCount = 0;
    unsigned nonBaseIndicesCount = 0;

    for (Operation *op : cluster.operations) {
      if (isa<openfhe::RotOp>(op)) {
        rotOpCount++;
      }
    }

    for (int64_t idx : cluster.rotationIndices) {
      if (std::find(baseSet.begin(), baseSet.end(), idx) == baseSet.end()) {
        nonBaseIndicesCount++;
      }
    }

    // Calculate the optimization score
    double score = rotOpCount * 2.0 + nonBaseIndicesCount * 3.0;
    if (cluster.distance > 0) {
      score += 10.0 / cluster.distance;
    }

    scoredClusters.push_back({i, score});
  }

  // Sort by score (descending)
  std::sort(
      scoredClusters.begin(), scoredClusters.end(),
      [](const std::pair<unsigned, double> &a,
         const std::pair<unsigned, double> &b) { return a.second > b.second; });

  // Extract cluster indices in priority order
  std::vector<unsigned> prioritizedIndices;
  for (const auto &pair : scoredClusters) {
    prioritizedIndices.push_back(pair.first);
  }

  return prioritizedIndices;
}

}  // anonymous namespace

#define GEN_PASS_DEF_ROTATIONDECOMPOSE
#include "lib/Transforms/RotationDecompose/RotationDecompose.h.inc"

struct RotationDecompose : impl::RotationDecomposeBase<RotationDecompose> {
  using RotationDecomposeBase::RotationDecomposeBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    // Extract all rotation indices from the IR
    std::set<int64_t> uniqueIndices;
    collectRotationIndices(op, uniqueIndices);

    if (uniqueIndices.empty()) {
      op->emitWarning() << "No rotation indices found in the IR";
      return;
    }

    // Convert set to vector for JSON
    std::vector<int64_t> rotationIndices(uniqueIndices.begin(),
                                         uniqueIndices.end());

    // Run the Python optimizer to find the optimal base set
    auto [baseSet, decompositions] = runPythonOptimizer(rotationIndices);

    if (baseSet.empty()) {
      op->emitError() << "Python optimizer failed to find a valid base set";
      return signalPassFailure();
    }

    // Apply the optimization by rewriting rotation operations
    if (failed(transformRotationOps(op, baseSet, decompositions))) {
      return signalPassFailure();
    }

    // Output some statistics
    LLVM_DEBUG(
        std::string optimizationMethod =
            useSimpleBsgs ? "simple BSGS" : "advanced MIP";
        llvm::dbgs() << "RotationDecompose Pass: Successfully optimized with "
                     << optimizationMethod << " using base set: ";
        for (int64_t idx : baseSet) { llvm::dbgs() << idx << " "; } llvm::dbgs()
        << "\n";);
    // op->print(llvm::outs(), OpPrintingFlags().printGenericOpForm());
  }

 private:
  // Collect all rotation indices from the IR
  void collectRotationIndices(Operation *op, std::set<int64_t> &indices) {
    // Walk through the IR and collect all rotation indices from
    // DeserializeKeyOp
    LLVM_DEBUG(llvm::dbgs() << "Scanning for rotation indices...\n";);
    op->walk([&](openfhe::DeserializeKeyOp deserOp) {
      if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
        int64_t index = indexAttr.getInt();
        if (indices.insert(index).second) {
          LLVM_DEBUG(llvm::dbgs()
                         << "Found rotation index: " << index << "\n";);
        }
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "Total unique rotation indices found: "
                            << indices.size() << "\n";);
  }

  // Create JSON input for Python optimizer
  std::string createJsonInput(const std::vector<int64_t> &indices) {
    std::stringstream ss;

    ss << "{\n";
    ss << "  \"target_indices\": [";
    for (size_t i = 0; i < indices.size(); ++i) {
      ss << indices[i];
      if (i < indices.size() - 1) {
        ss << ", ";
      }
    }
    ss << "],\n";
    ss << "  \"base_size\": " << baseSetSize << ",\n";
    ss << "  \"max_chain_length\": " << maxChainLength << ",\n";
    ss << "  \"include_non_targets\": true,\n";
    ss << "  \"saturated-bsgs\": true,\n";
    ss << "  \"time_limit\": " << timeLimit << "\n";
    ss << "}\n";

    return ss.str();
  }

  // Run the Python optimizer
  std::pair<std::vector<int64_t>, std::map<int64_t, std::vector<int64_t>>>
  runPythonOptimizer(const std::vector<int64_t> &indices) {
    // Create JSON input
    std::string jsonInput = createJsonInput(indices);

    // Write to temp file
    std::string inputFile = writeTempFile(jsonInput);
    std::string outputFile = inputFile + "_output";

    LLVM_DEBUG(llvm::dbgs() << "Current working directory: "
                            << std::filesystem::current_path() << "\n";);
    LLVM_DEBUG(llvm::dbgs() << "Python script path: " << pythonScript << "\n";);

    // Build command
    std::string cmd =
        ". /home/eymen/Documents/keymemrt_project/heir/.venv/bin/activate;  "
        "python3 " +
        pythonScript + " --input " + inputFile + " --output " + outputFile;

    if (verbose) {
      cmd += " --verbose";
      llvm::outs() << "Running command: " << cmd << "\n";
    }

    // Execute command
    std::string output;
    try {
      output = exec(cmd);
      if (verbose && !output.empty()) {
        llvm::outs() << "Command output: " << output << "\n";
      }
    } catch (const std::runtime_error &e) {
      llvm::errs() << "Error executing Python optimizer: " << e.what() << "\n";
      return {{}, {}};
    }

    // Read output file
    std::ifstream infile(outputFile);
    if (!infile) {
      llvm::errs() << "Failed to open output file: " << outputFile << "\n";
      return {{}, {}};
    }

    std::stringstream buffer;
    buffer << infile.rdbuf();
    std::string jsonOutput = buffer.str();

    // Clean up temp files
    std::remove(inputFile.c_str());
    std::remove(outputFile.c_str());

    // Parse JSON output
    return parseJsonOutput(jsonOutput, baseSetSize);
  }

  // Transform rotation operations based on the optimization results

  LogicalResult transformRotationOps(
      Operation *op, const std::vector<int64_t> &baseSet,
      const std::map<int64_t, std::vector<int64_t>> &decompositions) {
    // Track operations to delete
    llvm::SmallVector<Operation *, 16> opsToDelete;

    // First pass: Process deserialize operations
    op->walk([&](openfhe::DeserializeKeyOp deserOp) {
      if (!deserOp->hasAttr("index")) return;

      int64_t rotIndex = deserOp->getAttrOfType<IntegerAttr>("index").getInt();

      // Skip if this is a base index
      if (std::find(baseSet.begin(), baseSet.end(), rotIndex) !=
          baseSet.end()) {
        return;
      }

      // Get decomposition for this index
      auto decompIt = decompositions.find(rotIndex);
      if (decompIt == decompositions.end() || decompIt->second.empty()) {
        return;
      }

      const auto &baseIndices = decompIt->second;

      if (verbose) {
        llvm::outs() << "Will transform rotations with index " << rotIndex
                     << " using base indices: [";
        for (size_t i = 0; i < baseIndices.size(); ++i) {
          llvm::outs() << baseIndices[i];
          if (i < baseIndices.size() - 1) llvm::outs() << ", ";
        }
        llvm::outs() << "]\n";
      }

      // Create the base deserialize operations
      OpBuilder builder(deserOp);

      // Create deserialize ops for base indices
      llvm::DenseMap<int64_t, Value> baseKeyMap;
      for (int64_t baseIdx : baseIndices) {
        // Only create one deserialize op per unique base index
        if (baseKeyMap.count(baseIdx) > 0) continue;

        auto baseDeserOp = builder.create<openfhe::DeserializeKeyOp>(
            deserOp.getLoc(),
            openfhe::EvalKeyType::get(op->getContext(),
                                      builder.getIndexAttr(baseIdx)),
            deserOp.getCryptoContext());

        baseDeserOp->setAttr("index", builder.getIndexAttr(baseIdx));
        baseKeyMap[baseIdx] = baseDeserOp.getResult();
      }

      // Process all rotation operations that use this key
      llvm::SmallVector<openfhe::RotOp, 4> rotOps;
      for (Operation *user : deserOp.getResult().getUsers()) {
        if (auto rotOp = dyn_cast<openfhe::RotOp>(user)) {
          rotOps.push_back(rotOp);
        }
      }

      for (auto rotOp : rotOps) {
        // Replace with a chain of rotations using base keys
        builder.setInsertionPoint(rotOp);

        Value cryptoContext = rotOp.getCryptoContext();
        Value inputCiphertext = rotOp.getCiphertext();
        Type resultType = rotOp.getType();

        // Chain rotations one after another - start with the input
        Value currentResult = inputCiphertext;

        // Create a rotation for each base index in the decomposition
        for (int64_t baseIdx : baseIndices) {
          Value baseKey = baseKeyMap[baseIdx];

          auto baseRotOp = builder.create<openfhe::RotOp>(
              rotOp.getLoc(), resultType, cryptoContext, currentResult,
              baseKey);

          // Update current result for the next rotation in chain
          currentResult = baseRotOp.getResult();
        }

        // Replace original rotation with our chain result
        rotOp.getResult().replaceAllUsesWith(currentResult);

        // Mark for deletion
        opsToDelete.push_back(rotOp);
      }

      // Find and process clear operation
      for (Operation *user : deserOp.getResult().getUsers()) {
        if (auto clearOp = dyn_cast<openfhe::ClearKeyOp>(user)) {
          // Create clear operations for base keys
          builder.setInsertionPoint(clearOp);

          for (auto &[baseIdx, baseKey] : baseKeyMap) {
            builder.create<openfhe::ClearKeyOp>(
                clearOp.getLoc(), clearOp.getCryptoContext(), baseKey);
          }

          // Mark for deletion
          opsToDelete.push_back(clearOp);
          break;
        }
      }

      // Mark deserialize op for deletion
      opsToDelete.push_back(deserOp);
    });

    // Delete all marked operations
    for (Operation *op : opsToDelete) {
      op->erase();
    }

    return success();
  }
};
}  // namespace heir
}  // namespace mlir
