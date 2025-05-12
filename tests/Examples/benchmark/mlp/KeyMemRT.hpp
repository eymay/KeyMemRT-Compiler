#ifndef KEY_MANAGER_H_
#define KEY_MANAGER_H_

#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>

#include "cryptocontext.h"
#include "openfhe.h"

using namespace lbcrypto;

enum class KeyMemMode {
  IGNORE,     // Ignores all key management operations
  IMPERATIVE  // Performs operations as requested
};

inline std::string getModeString(KeyMemMode mode) {
  switch (mode) {
    case KeyMemMode::IGNORE:
      return "ignore";
    case KeyMemMode::IMPERATIVE:
      return "imperative";
  }
  return "unknown";
}

// Simple CLI helper that can be reused across benchmarks
class BenchmarkCLI {
 public:
  // Parse command line arguments
  static void parseArgs(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
      std::string arg = argv[i];

      if (arg == "--key-mode" && i + 1 < argc) {
        std::string mode = argv[++i];
        if (mode == "ignore") {
          keymem_mode = KeyMemMode::IGNORE;
        } else if (mode == "imperative") {
          keymem_mode = KeyMemMode::IMPERATIVE;
        }
        std::cout << "KeyMemRT mode set to: " << getModeString(keymem_mode)
                  << std::endl;
      } else if (arg == "--output-base" && i + 1 < argc) {
        output_base = argv[++i];
      } else if (arg == "--help" || arg == "-h") {
        printHelp();
      }
    }
  }

  static void printHelp() {
    std::cout
        << "Benchmark CLI Options:\n"
        << "  --key-mode <mode>    : Set key memory mode (ignore|imperative)\n"
        << "  --output-base <name> : Base name for output files\n"
        << "  --help, -h           : Display this help message\n"
        << std::endl;
  }

  // Generate output filename based on benchmark name and mode
  static std::string getOutputFilename(const std::string& benchmark_name) {
    std::string filename =
        output_base.empty() ? "resource_usage_" + benchmark_name : output_base;

    // Add mode suffix if not already present
    std::string mode_str = getModeString(keymem_mode);
    if (filename.find(mode_str) == std::string::npos) {
      filename += "_" + mode_str;
    }

    return filename;
  }

  // Gets the key memory mode
  static KeyMemMode getKeyMemMode() { return keymem_mode; }

 private:
  static inline KeyMemMode keymem_mode;
  static inline std::string output_base;
};

class KeyMemRT {
 public:
  KeyMemRT(CryptoContext<DCRTPoly> context,
           KeyMemMode mode = KeyMemMode::IMPERATIVE)
      : cc(std::move(context)), keyTag(""), operationMode(mode) {}

  KeyMemRT() : cc(nullptr), keyTag(""), operationMode(KeyMemMode::IMPERATIVE) {}

  // Initialize from CLI arguments
  void initFromArgs(int argc, char* argv[]) {
    // Use the CLI helper to parse settings
    BenchmarkCLI::parseArgs(argc, argv);
    operationMode = BenchmarkCLI::getKeyMemMode();
  }

  void setCryptoContext(CryptoContext<DCRTPoly>& context) { cc = context; }

  void setKeyTag(const std::string& tag) { keyTag = tag; }

  void setKeyMemMode(KeyMemMode mode) { operationMode = mode; }

  KeyMemMode getOperationMode() const { return operationMode; }

  void setRotIndices(const std::vector<int32_t>& indices) {
    rotIndices = indices;
  }

  bool serializeKey(int rotationIndex) {
    if (operationMode == KeyMemMode::IGNORE) {
      return true;
    }

    auto automorphismIndex = getAutomorphismIndex(rotationIndex);
    std::ofstream keyFile(getKeyFilename(rotationIndex), std::ios::binary);
    if (!keyFile) {
      return false;
    }
    return cc->SerializeEvalAutomorphismKey(keyFile, SerType::BINARY, keyTag,
                                            {automorphismIndex});
  }

  bool deserializeKey(int rotationIndex) {
    if (operationMode == KeyMemMode::IGNORE) {
      return true;
    }

    auto automorphismIndex = getAutomorphismIndex(rotationIndex);
    std::ifstream keyFile(getKeyFilename(rotationIndex), std::ios::binary);
    if (!keyFile) {
      return false;
    }

    bool success = cc->DeserializeEvalAutomorphismKey(
        keyFile, SerType::BINARY, keyTag, {automorphismIndex});

    // Track loaded keys
    if (success) {
      loadedKeys.insert(rotationIndex);
    }

    return success;
  }

  bool clearKey(int rotationIndex) {
    if (operationMode == KeyMemMode::IGNORE) {
      return true;
    }

    auto automorphismIndex = getAutomorphismIndex(rotationIndex);
    auto keyMap = cc->GetEvalAutomorphismKeyMap(keyTag);
    keyMap.erase(automorphismIndex);
    loadedKeys.erase(rotationIndex);
    return true;
  }

  bool serializeAllKeys() {
    if (operationMode == KeyMemMode::IGNORE) {
      return true;
    }

    bool success = true;
    for (const auto& rotIndex : rotIndices) {
      success &= serializeKey(rotIndex);
    }
    return success;
  }

  bool deserializeAllKeys() {
    if (operationMode == KeyMemMode::IGNORE) {
      return true;
    }

    bool success = true;
    for (const auto& rotIndex : rotIndices) {
      success &= deserializeKey(rotIndex);
    }
    return success;
  }

  bool clearAllKeys() {
    if (operationMode == KeyMemMode::IGNORE) {
      return true;
    }

    cc->ClearEvalAutomorphismKeys(keyTag);
    loadedKeys.clear();
    return true;
  }

  std::string getKeyFilename(int rotationIndex) const {
    return "rotation_key_" + std::to_string(rotationIndex) + ".bin";
  }

  bool checkKeyExists(int rotationIndex) const {
    std::ifstream keyFile(getKeyFilename(rotationIndex));
    return keyFile.good();
  }

  // Print stats about key operations
  void printKeyStats() const {
    std::cout << "KeyMemRT Stats:\n"
              << "  Mode: " << getModeString(operationMode) << "\n"
              << "  Total rotation indices: " << rotIndices.size() << "\n"
              << "  Keys loaded: " << loadedKeys.size() << "\n";
  }

 private:
  usint getAutomorphismIndex(usint rotationIndex) const {
    return cc->FindAutomorphismIndex(rotationIndex);
  }

  CryptoContext<DCRTPoly> cc;
  std::string keyTag;
  std::vector<int32_t> rotIndices;
  KeyMemMode operationMode;
  std::set<int32_t> loadedKeys;
};
#endif  // KEY_MANAGER_H_
