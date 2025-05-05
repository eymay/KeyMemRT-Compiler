#include "lib/Transforms/RotationDecompose/RotationDecompose.h"

#include <array>
#include <cstdio>
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
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/IRMapping.h"
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"

// Include for getting process ID
#include <unistd.h>

namespace mlir {
namespace heir {

constexpr bool verbose = true;

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
  if (baseSet.size() != baseSetSize) {
    llvm::errs() << "Warning: Expected base set size " << baseSetSize
                 << " but got " << baseSet.size() << "\n";
  }

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
    llvm::outs()
        << "RotationDecompose Pass: Successfully optimized with base set: ";
    for (int64_t idx : baseSet) {
      llvm::outs() << idx << " ";
    }
    llvm::outs() << "\n";
    op->print(llvm::outs(), OpPrintingFlags().printGenericOpForm());
  }

 private:
  // Collect all rotation indices from the IR
  void collectRotationIndices(Operation *op, std::set<int64_t> &indices) {
    // Walk through the IR and collect all rotation indices from
    // DeserializeKeyOp
    llvm::outs() << "Scanning for rotation indices...\n";
    op->walk([&](openfhe::DeserializeKeyOp deserOp) {
      if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
        int64_t index = indexAttr.getInt();
        if (indices.insert(index).second) {
          llvm::outs() << "Found rotation index: " << index << "\n";
        }
      }
    });

    llvm::outs() << "Total unique rotation indices found: " << indices.size()
                 << "\n";
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

    // Run Python optimizer
    std::string cmd = "python " + pythonScript + " --input " + inputFile +
                      " --output " + outputFile;

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
    // Track operations to delete and replace
    llvm::SmallVector<Operation *, 16> opsToDelete;

    // Structure to represent a cluster of rotation operations
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

    // Find clusters of rotations
    llvm::SmallVector<RotationCluster, 8> clusters;

    // Helper to get index of an operation in its parent block
    auto getOpIndexInBlock = [](Operation *op) -> unsigned {
      Block *block = op->getBlock();
      unsigned idx = 0;
      for (auto &blockOp : block->getOperations()) {
        if (&blockOp == op) {
          return idx;
        }
        idx++;
      }
      return UINT_MAX;  // Not found
    };

    // Walk through all blocks in the module
    op->walk([&](Block *block) {
      // Skip empty blocks
      if (block->empty()) {
        return;
      }

      // Find all rotation operations in this block
      llvm::SmallVector<openfhe::RotOp, 16> rotOps;
      llvm::SmallVector<unsigned, 16> rotOpIndices;

      unsigned idx = 0;
      for (auto &blockOp : block->getOperations()) {
        if (auto rotOp = dyn_cast<openfhe::RotOp>(&blockOp)) {
          rotOps.push_back(rotOp);
          rotOpIndices.push_back(idx);
        }
        idx++;
      }

      // Skip blocks with fewer than 2 rotation operations
      if (rotOps.size() < 2) {
        return;
      }

      // Identify clusters of rotations based on distance
      const unsigned maxDistance = 10;  // Configure this based on your needs

      // Analyze distances between rotations
      for (unsigned i = 0; i < rotOps.size() - 1; i++) {
        RotationCluster cluster;
        cluster.startIdx = rotOpIndices[i];
        cluster.endIdx = rotOpIndices[i];
        cluster.distance = 0;

        // Start a new cluster with this rotation
        auto startRotOp = rotOps[i];

        // Add the rotation and its associated deserialize/clear ops
        cluster.operations.push_back(startRotOp);

        // Get rotation index
        int64_t rotIndex = -1;
        auto evalKey = startRotOp.getEvalKey();
        if (auto deserOp = dyn_cast_or_null<openfhe::DeserializeKeyOp>(
                evalKey.getDefiningOp())) {
          if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
            rotIndex = indexAttr.getInt();
            cluster.rotationIndices.insert(rotIndex);
            cluster.operations.push_back(deserOp);

            // Find the associated clear op
            for (Operation *user : deserOp.getResult().getUsers()) {
              if (auto clearOp = dyn_cast<openfhe::ClearKeyOp>(user)) {
                cluster.operations.push_back(clearOp);
                break;
              }
            }
          }
        }

        // Try to extend this cluster
        for (unsigned j = i + 1; j < rotOps.size(); j++) {
          unsigned distance = rotOpIndices[j] - rotOpIndices[j - 1];

          // If the rotation is too far, end this cluster
          if (distance > maxDistance) {
            break;
          }

          // Add this rotation to the current cluster
          auto nextRotOp = rotOps[j];
          cluster.operations.push_back(nextRotOp);
          cluster.endIdx = rotOpIndices[j];
          cluster.distance += distance;

          // Get rotation index
          auto nextEvalKey = nextRotOp.getEvalKey();
          if (auto deserOp = dyn_cast_or_null<openfhe::DeserializeKeyOp>(
                  nextEvalKey.getDefiningOp())) {
            if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
              int64_t nextRotIndex = indexAttr.getInt();
              cluster.rotationIndices.insert(nextRotIndex);
              cluster.operations.push_back(deserOp);

              // Find the associated clear op
              for (Operation *user : deserOp.getResult().getUsers()) {
                if (auto clearOp = dyn_cast<openfhe::ClearKeyOp>(user)) {
                  cluster.operations.push_back(clearOp);
                  break;
                }
              }
            }
          }
        }

        // Only keep clusters with multiple different rotation indices
        if (cluster.rotationIndices.size() > 1) {
          clusters.push_back(cluster);
        }
      }
    });

    if (clusters.empty()) {
      if (verbose) {
        llvm::outs()
            << "No suitable rotation clusters found for optimization\n";
      }
      return success();
    }

    if (verbose) {
      llvm::outs() << "Found " << clusters.size() << " rotation clusters\n";
      for (unsigned i = 0; i < clusters.size(); i++) {
        llvm::outs() << "Cluster " << i << ": " << clusters[i].operations.size()
                     << " operations, distance " << clusters[i].distance
                     << ", rotation indices: ";
        for (auto idx : clusters[i].rotationIndices) {
          llvm::outs() << idx << " ";
        }
        llvm::outs() << "\n";
      }
    }

    // For each cluster, we need to decide whether to apply the transformation
    for (auto &cluster : clusters) {
      // Skip clusters where all indices are already in the base set
      bool allInBaseSet = true;
      for (int64_t idx : cluster.rotationIndices) {
        if (std::find(baseSet.begin(), baseSet.end(), idx) == baseSet.end()) {
          allInBaseSet = false;
          break;
        }
      }

      if (allInBaseSet) {
        if (verbose) {
          llvm::outs()
              << "Skipping cluster with all indices already in base set\n";
        }
        continue;
      }

      // Process the cluster
      if (verbose) {
        llvm::outs() << "Processing cluster with indices: ";
        for (auto idx : cluster.rotationIndices) {
          llvm::outs() << idx << " ";
        }
        llvm::outs() << "\n";
      }

      // First find all non-base deserialize operations in this cluster
      struct DeserializedKey {
        openfhe::DeserializeKeyOp deserOp;
        openfhe::ClearKeyOp clearOp;
        int64_t rotIndex;
        std::vector<int64_t> baseIndices;  // Base indices from decomposition
        std::vector<Value> baseKeys;       // Corresponding base key values
      };

      llvm::SmallVector<DeserializedKey, 16> keysToDecompose;

      for (Operation *op : cluster.operations) {
        auto deserOp = dyn_cast<openfhe::DeserializeKeyOp>(op);
        if (!deserOp) {
          continue;
        }

        if (auto indexAttr = deserOp->getAttrOfType<IntegerAttr>("index")) {
          int64_t index = indexAttr.getInt();

          // Skip base indices, we keep these as-is
          if (std::find(baseSet.begin(), baseSet.end(), index) !=
              baseSet.end()) {
            continue;
          }

          // Find the associated clear op if any
          openfhe::ClearKeyOp clearOp = nullptr;
          for (Operation *user : deserOp.getResult().getUsers()) {
            if (auto op = dyn_cast<openfhe::ClearKeyOp>(user)) {
              clearOp = op;
              break;
            }
          }

          // Get the decomposition for this index
          auto decompIt = decompositions.find(index);
          if (decompIt == decompositions.end() || decompIt->second.empty()) {
            llvm::errs() << "No decomposition found for index " << index
                         << "\n";
            continue;
          }

          // Record for processing
          keysToDecompose.push_back(
              {deserOp, clearOp, index, decompIt->second, {}});
        }
      }

      if (keysToDecompose.empty()) {
        if (verbose) {
          llvm::outs() << "No operations to transform in this cluster\n";
        }
        continue;
      }

      // Process each deserialize operation
      OpBuilder builder(op->getContext());

      for (auto &keyInfo : keysToDecompose) {
        if (verbose) {
          llvm::outs() << "Processing deserialize op for index "
                       << keyInfo.rotIndex << " using base indices: [";
          for (size_t i = 0; i < keyInfo.baseIndices.size(); ++i) {
            llvm::outs() << keyInfo.baseIndices[i];
            if (i < keyInfo.baseIndices.size() - 1) llvm::outs() << ", ";
          }
          llvm::outs() << "]\n";
        }

        // Create deserialize operations for each base index
        builder.setInsertionPoint(keyInfo.deserOp);

        for (int64_t baseIdx : keyInfo.baseIndices) {
          auto baseDeserOp = builder.create<openfhe::DeserializeKeyOp>(
              keyInfo.deserOp.getLoc(),
              openfhe::EvalKeyType::get(keyInfo.deserOp.getContext(),
                                        builder.getIndexAttr(baseIdx)),
              keyInfo.deserOp.getCryptoContext());

          baseDeserOp->setAttr("index", builder.getIndexAttr(baseIdx));
          keyInfo.baseKeys.push_back(baseDeserOp.getResult());
        }

        // Now find and transform all rotation operations that use this key
        llvm::SmallVector<openfhe::RotOp, 8> rotOpsToTransform;

        for (Operation *user : keyInfo.deserOp.getResult().getUsers()) {
          if (auto rotOp = dyn_cast<openfhe::RotOp>(user)) {
            rotOpsToTransform.push_back(rotOp);
          }
        }

        for (auto rotOp : rotOpsToTransform) {
          // Create the decomposition sequence for this rotation
          builder.setInsertionPoint(rotOp);

          // Get the input values
          Value cryptoContext = rotOp.getCryptoContext();
          Value inputCiphertext = rotOp.getCiphertext();
          Type resultType = rotOp.getType();
          Location loc = rotOp.getLoc();

          // Create the first rotation with the first base key
          auto firstRotOp = builder.create<openfhe::RotOp>(
              loc, resultType, cryptoContext, inputCiphertext,
              keyInfo.baseKeys[0]);

          // Current result starts with the first rotation
          Value currentResult = firstRotOp.getResult();

          // Add the rest of the rotations
          for (size_t i = 1; i < keyInfo.baseKeys.size(); ++i) {
            // Create a rotation with this base key
            auto baseRotOp = builder.create<openfhe::RotOp>(
                loc, resultType, cryptoContext, inputCiphertext,
                keyInfo.baseKeys[i]);

            // Add to the current result
            currentResult = builder.create<openfhe::AddOp>(
                loc, resultType, cryptoContext, currentResult,
                baseRotOp.getResult());
          }

          // Replace the original rotation's result with our sequence result
          rotOp.getResult().replaceAllUsesWith(currentResult);

          // Mark the original rotation for deletion
          opsToDelete.push_back(rotOp);
        }

        // If there's a clear op, replace it with clears for all base keys
        if (keyInfo.clearOp) {
          builder.setInsertionPoint(keyInfo.clearOp);

          // Create clear operations for each base key
          for (Value baseKey : keyInfo.baseKeys) {
            builder.create<openfhe::ClearKeyOp>(
                keyInfo.clearOp.getLoc(), keyInfo.clearOp.getCryptoContext(),
                baseKey);
          }

          // Mark the original clear op for deletion
          opsToDelete.push_back(keyInfo.clearOp);
        }

        // Mark the original deserialize op for deletion
        opsToDelete.push_back(keyInfo.deserOp);
      }
    }

    // Delete all marked operations
    for (Operation *op : opsToDelete) {
      op->erase();
    }

    return success();
  };
};
}  // namespace heir
}  // namespace mlir
