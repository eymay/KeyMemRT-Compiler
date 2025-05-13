#ifndef KEY_MANAGER_H_
#define KEY_MANAGER_H_

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>

#include "cryptocontext.h"
#include "openfhe.h"

#ifndef KEYMEMRT_LOGGER_H_
#define KEYMEMRT_LOGGER_H_

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

using namespace lbcrypto;

/**
 * Compresses evaluation/rotation keys to the specified level
 * @param ek Map of evaluation keys to compress
 * @param level Number of RNS components to drop from the end (if positive)
 *        OR target level to keep (if negative)
 */
void CompressEvalKeysToLevel(std::map<usint, EvalKey<DCRTPoly>>& ek,
                             int level) {
  for (auto& keyPair : ek) {
    auto evalKey = keyPair.second;

    // Skip if the key is null
    if (!evalKey) continue;

    // Get the current A and B vectors
    std::vector<DCRTPoly> a = evalKey->GetAVector();
    std::vector<DCRTPoly> b = evalKey->GetBVector();

    // Skip empty keys
    if (a.empty() || b.empty()) continue;

    // Get the current size (number of towers)
    size_t currentSize = a[0].GetParams()->GetParams().size();

    // Calculate how many elements to drop
    size_t elementsToDrop = 0;

    if (level >= 0) {
      // Level is positive: directly specifies number of elements to drop
      elementsToDrop = level;
    } else {
      // Level is negative: specifies target level to keep
      int targetLevel = -level;
      if (targetLevel > 0 && static_cast<size_t>(targetLevel) < currentSize) {
        elementsToDrop = currentSize - targetLevel;
      }
    }

    // Safety check: don't drop all towers
    if (elementsToDrop >= currentSize) {
      std::cout << "Cannot drop " + std::to_string(elementsToDrop) +
                       " elements from a key with only " +
                       std::to_string(currentSize) +
                       " towers. Skipping compression.";
      continue;
    }

    // Skip if no elements to drop
    if (elementsToDrop == 0) continue;

    std::cout << "Compressing key: dropping " + std::to_string(elementsToDrop) +
                     " elements from " + std::to_string(currentSize) +
                     " towers";

    // Perform compression by dropping elements
    for (size_t k = 0; k < a.size(); k++) {
      a[k].DropLastElements(elementsToDrop);
      b[k].DropLastElements(elementsToDrop);
    }

    // Update the key with compressed vectors
    evalKey->ClearKeys();
    evalKey->SetAVector(std::move(a));
    evalKey->SetBVector(std::move(b));
  }
}

enum class LogLevel {
  DEBUG,
  INFO,
  WARNING,
  ERROR,
  OFF  // No logging
};

class Logger {
 public:
  static Logger& getInstance() {
    static Logger instance;
    return instance;
  }

  void setLogLevel(LogLevel level) { logLevel = level; }

  bool isDebugEnabled() const { return logLevel <= LogLevel::DEBUG; }

  bool isInfoEnabled() const { return logLevel <= LogLevel::INFO; }

  bool isWarningEnabled() const { return logLevel <= LogLevel::WARNING; }

  bool isErrorEnabled() const { return logLevel <= LogLevel::ERROR; }

  std::string levelToString(LogLevel level) {
    switch (level) {
      case LogLevel::DEBUG:
        return "DEBUG";
      case LogLevel::INFO:
        return "INFO";
      case LogLevel::WARNING:
        return "WARNING";
      case LogLevel::ERROR:
        return "ERROR";
      case LogLevel::OFF:
        return "OFF";
      default:
        return "UNKNOWN";
    }
  }

  void setLogToFile(bool enable, const std::string& filename = "keymemrt.log") {
    std::lock_guard<std::mutex> lock(logMutex);
    if (enable && !logFile.is_open()) {
      logFile.open(filename, std::ios::out | std::ios::app);
      if (logFile.is_open() && logLevel < LogLevel::OFF) {
        logFile << "\n[" << getTimestamp() << "] [INFO] Logger started"
                << std::endl;
      }
    } else if (!enable && logFile.is_open()) {
      if (logLevel < LogLevel::OFF) {
        logFile << "[" << getTimestamp() << "] [INFO] Logger stopped"
                << std::endl;
      }
      logFile.close();
    }
    logToFile = enable;
  }

  void setLogToConsole(bool enable) { logToConsole = enable; }

  template <typename... Args>
  void debug(const std::string& format, Args... args) {
    if (logLevel <= LogLevel::DEBUG) {
      log(LogLevel::DEBUG, format, args...);
    }
  }

  template <typename... Args>
  void info(const std::string& format, Args... args) {
    if (logLevel <= LogLevel::INFO) {
      log(LogLevel::INFO, format, args...);
    }
  }

  template <typename... Args>
  void warning(const std::string& format, Args... args) {
    if (logLevel <= LogLevel::WARNING) {
      log(LogLevel::WARNING, format, args...);
    }
  }

  template <typename... Args>
  void error(const std::string& format, Args... args) {
    if (logLevel <= LogLevel::ERROR) {
      log(LogLevel::ERROR, format, args...);
    }
  }

 private:
  Logger() : logLevel(LogLevel::INFO), logToConsole(true), logToFile(false) {}
  ~Logger() {
    if (logFile.is_open()) {
      logFile.close();
    }
  }

  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

  std::string getTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) %
                  1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << now_ms.count();
    return ss.str();
  }

  template <typename... Args>
  void log(LogLevel level, const std::string& format, Args... args) {
    std::lock_guard<std::mutex> lock(logMutex);

    std::string timestamp = getTimestamp();
    std::string levelStr = levelToString(level);

    // Format the message with the provided arguments
    std::string message = formatString(format, args...);

    // Create the full log entry
    std::stringstream logEntry;
    logEntry << "[" << timestamp << "] [" << levelStr << "] " << message;

    // Output to console if enabled
    if (logToConsole) {
      std::cout << logEntry.str() << std::endl;
    }

    // Output to file if enabled
    if (logToFile && logFile.is_open()) {
      logFile << logEntry.str() << std::endl;
    }
  }

  // Simple string formatter that replaces {} placeholders with arguments
  template <typename T, typename... Args>
  std::string formatString(const std::string& format, T value, Args... args) {
    size_t openBracePos = format.find("{}");
    if (openBracePos == std::string::npos) {
      return format;
    }

    std::stringstream ss;
    ss << value;

    return format.substr(0, openBracePos) + ss.str() +
           formatString(format.substr(openBracePos + 2), args...);
  }

  // Base case for the recursion
  std::string formatString(const std::string& format) { return format; }

  LogLevel logLevel;
  bool logToConsole;
  bool logToFile;
  std::ofstream logFile;
  std::mutex logMutex;
};

#define LOG_DEBUG(format, ...) \
  Logger::getInstance().debug(format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) Logger::getInstance().info(format, ##__VA_ARGS__)
#define LOG_WARNING(format, ...) \
  Logger::getInstance().warning(format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) \
  Logger::getInstance().error(format, ##__VA_ARGS__)

#endif  // KEYMEMRT_LOGGER_H_"

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
    // Default params
    ser_force = false;
    ser_single_file = false;
    log_level = LogLevel::INFO;
    log_to_file = false;
    log_filename = "keymemrt.log";

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
      } else if (arg == "--ser-force") {
        ser_force = true;
        std::cout << "Force serialization enabled" << std::endl;
      } else if (arg == "--ser-single-file") {
        ser_single_file = true;
        std::cout << "Single file serialization enabled" << std::endl;
      } else if (arg == "--log-level" && i + 1 < argc) {
        std::string level = argv[++i];
        if (level == "debug") {
          log_level = LogLevel::DEBUG;
        } else if (level == "info") {
          log_level = LogLevel::INFO;
        } else if (level == "warning") {
          log_level = LogLevel::WARNING;
        } else if (level == "error") {
          log_level = LogLevel::ERROR;
        } else if (level == "off") {
          log_level = LogLevel::OFF;
        }
        std::cout << "Log level set to: " << level << std::endl;
      } else if (arg == "--log-file" && i + 1 < argc) {
        log_filename = argv[++i];
        log_to_file = true;
        std::cout << "Logging to file: " << log_filename << std::endl;
      } else if (arg == "--log-console-off") {
        log_to_console = false;
        std::cout << "Console logging disabled" << std::endl;
      } else if (arg == "--help" || arg == "-h") {
        printHelp();
      }
    }

    // Initialize logger based on parameters
    Logger::getInstance().setLogLevel(log_level);
    if (log_to_file) {
      Logger::getInstance().setLogToFile(true, log_filename);
    }
    Logger::getInstance().setLogToConsole(log_to_console);

    LOG_INFO("BenchmarkCLI initialized with mode: {}",
             getModeString(keymem_mode));
    if (ser_force) LOG_INFO("Force serialization enabled");
    if (ser_single_file) LOG_INFO("Single file serialization enabled");
  }

  static void printHelp() {
    std::cout
        << "Benchmark CLI Options:\n"
        << "  --key-mode <mode>    : Set key memory mode (ignore|imperative)\n"
        << "  --output-base <name> : Base name for output files\n"
        << "  --ser-force          : Force serialization even in ignore mode\n"
        << "  --ser-single-file    : Use single file for all keys\n"
        << "  --log-level <level>  : Set log level "
           "(debug|info|warning|error|off)\n"
        << "  --log-file <file>    : Enable logging to specified file\n"
        << "  --log-console-off    : Disable console logging\n"
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

    LOG_DEBUG("Generated output filename: {}", filename);
    return filename;
  }

  // Gets the key memory mode
  static KeyMemMode getKeyMemMode() { return keymem_mode; }
  static bool getSerForce() { return ser_force; }
  static bool getSerSingleFile() { return ser_single_file; }
  static LogLevel getLogLevel() { return log_level; }
  static std::string getLogFilename() { return log_filename; }

 private:
  static inline KeyMemMode keymem_mode;
  static inline std::string output_base;
  static inline bool ser_force;
  static inline bool ser_single_file;
  static inline LogLevel log_level;
  static inline bool log_to_file;
  static inline bool log_to_console = true;
  static inline std::string log_filename;
};

class KeyMemRT {
 public:
  KeyMemRT(CryptoContext<DCRTPoly> context,
           KeyMemMode mode = KeyMemMode::IMPERATIVE)
      : cc(std::move(context)), keyTag(""), operationMode(mode) {
    LOG_INFO("KeyMemRT initialized with mode: {}", getModeString(mode));
  }

  KeyMemRT() : cc(nullptr), keyTag(""), operationMode(KeyMemMode::IMPERATIVE) {
    LOG_INFO("KeyMemRT default initialized with mode: {}",
             getModeString(operationMode));
  }

  // Initialize from CLI arguments
  void initFromArgs(int argc, char* argv[]) {
    LOG_INFO("Initializing KeyMemRT from command line arguments");
    // Use the CLI helper to parse settings
    BenchmarkCLI::parseArgs(argc, argv);
    operationMode = BenchmarkCLI::getKeyMemMode();
    LOG_INFO("KeyMemRT mode set to: {}", getModeString(operationMode));
  }

  void setCryptoContext(CryptoContext<DCRTPoly>& context) {
    cc = context;
    LOG_INFO("Crypto context set");
  }

  void setKeyTag(const std::string& tag) {
    keyTag = tag;
    LOG_INFO("Key tag set to: {}", tag);
  }

  void setKeyMemMode(KeyMemMode mode) {
    LOG_INFO("Changing mode from {} to {}", getModeString(operationMode),
             getModeString(mode));
    operationMode = mode;
  }

  KeyMemMode getOperationMode() const { return operationMode; }

  void setRotIndices(const std::vector<int32_t>& indices) {
    rotIndices = indices;
    LOG_INFO("Rotation indices set: {} indices", indices.size());
    if (Logger::getInstance().isDebugEnabled()) {
      std::stringstream ss;
      for (size_t i = 0; i < std::min(indices.size(), size_t(10)); i++) {
        ss << indices[i];
        if (i < std::min(indices.size(), size_t(10)) - 1) ss << ", ";
      }
      if (indices.size() > 10) ss << "...";
      LOG_DEBUG("Sample indices: {}", ss.str());
    }
  }

  bool serializeKeysAtDepth(const std::vector<int32_t>& indices, int depth) {
    if (operationMode == KeyMemMode::IGNORE) {
      LOG_DEBUG("IGNORE mode: Skipping serialize keys at depth {}", depth);
      return true;
    }

    LOG_INFO("Serializing keys at depth {}", depth);

    // Important: Get keys BEFORE attempting to compress them
    auto allKeys = cc->GetEvalAutomorphismKeyMap(keyTag);

    // If no keys are found, we can't proceed
    if (allKeys.empty()) {
      LOG_ERROR("No evaluation keys found for key tag [{}]", keyTag);
      return false;
    }

    // Now compress the keys
    bool success = true;

    // Now compress and serialize each key individually
    for (int32_t rotIndex : indices) {
      auto automorphismIndex = cc->FindAutomorphismIndex(rotIndex);

      // Verify the key exists before trying to compress/serialize it
      if (allKeys.find(automorphismIndex) == allKeys.end()) {
        LOG_ERROR("Key not found for rotation index {} (automorphism {})",
                  rotIndex, automorphismIndex);
        success = false;
        continue;
      }

      // Create a map with just this one key
      std::map<usint, EvalKey<DCRTPoly>> keyToSerialize;
      keyToSerialize[automorphismIndex] = allKeys[automorphismIndex];

      // Compress just this key
      CompressEvalKeysToLevel(keyToSerialize, depth);

      // Now serialize the compressed key
      std::string filename = getKeyFilename(rotIndex, depth);
      LOG_INFO(
          "Serializing key for rotation index {} (automorphism {}) at depth {} "
          "to {}",
          rotIndex, automorphismIndex, depth, filename);

      std::ofstream keyFile(filename, std::ios::binary);
      if (!keyFile) {
        LOG_ERROR("Failed to open file for writing: {}", filename);
        success = false;
        continue;
      }

      bool result = cc->SerializeEvalAutomorphismKey(
          keyFile, SerType::BINARY, keyTag, {automorphismIndex});

      if (result) {
        LOG_INFO(
            "Successfully serialized key for rotation index {} at depth {}",
            rotIndex, depth);
      } else {
        LOG_ERROR("Failed to serialize key for rotation index {} at depth {}",
                  rotIndex, depth);
        success = false;
      }
    }

    return success;
  }

  bool deserializeKey(int rotationIndex, int keyDepth) {
    if (operationMode == KeyMemMode::IGNORE) {
      LOG_DEBUG("IGNORE mode: Skipping deserialize for rotation index {}",
                rotationIndex);
      return true;
    }

    auto automorphismIndex = getAutomorphismIndex(rotationIndex);

    std::string filename;

    filename = getKeyFilename(rotationIndex, keyDepth);
    std::ifstream testFile(filename);
    if (!testFile.good()) {
      LOG_ERROR("Could not find any depth file for rotation index {}",
                rotationIndex);
      return false;
    }

    LOG_INFO("Deserializing key for rotation index {} at depth {} from {}",
             rotationIndex, keyDepth, filename);

    std::ifstream keyFile(filename, std::ios::binary);
    if (!keyFile) {
      LOG_ERROR("Failed to open file for reading: {}", filename);
      return false;
    }

    bool success = cc->DeserializeEvalAutomorphismKey(
        keyFile, SerType::BINARY, keyTag, {automorphismIndex});

    if (success) {
      LOG_INFO(
          "Successfully deserialized key for rotation index {} at depth {}",
          rotationIndex, keyDepth);
      loadedKeys.insert(rotationIndex);
    } else {
      LOG_ERROR("Failed to deserialize key for rotation index {} at depth {}",
                rotationIndex, keyDepth);
    }

    return success;
  }

  // Add overload for backward compatibility
  bool serializeKey(int rotationIndex) {
    return serializeKeysAtDepth({rotationIndex}, 0);
  }
  bool deserializeKey(int rotationIndex) {
    return deserializeKey(rotationIndex, 0);
  }

  // Helper to get depth-specific filename
  std::string getKeyFilename(int rotationIndex, int depth) const {
    if (depth == 0) {
      return "/tmp/rotation_key_" + std::to_string(rotationIndex) + ".bin";
    } else {
      return "/tmp/rotation_key_" + std::to_string(rotationIndex) + "_d" +
             std::to_string(depth) + ".bin";
    }
  }

  bool clearKey(int rotationIndex) {
    if (operationMode == KeyMemMode::IGNORE) {
      LOG_DEBUG("IGNORE mode: Skipping clear for rotation index {}",
                rotationIndex);
      return true;
    }

    auto automorphismIndex = getAutomorphismIndex(rotationIndex);
    LOG_INFO("Clearing key for rotation index {} (automorphism {})",
             rotationIndex, automorphismIndex);

    auto keyMap = cc->GetEvalAutomorphismKeyMap(keyTag);
    auto sizeBefore = keyMap.size();
    keyMap.erase(automorphismIndex);
    auto sizeAfter = keyMap.size();

    bool success = (sizeBefore != sizeAfter);
    if (success) {
      LOG_INFO("Successfully cleared key for rotation index {}", rotationIndex);
      loadedKeys.erase(rotationIndex);
    } else {
      LOG_WARNING("Key for rotation index {} not found or couldn't be cleared",
                  rotationIndex);
    }
    return true;  // Always return true for backward compatibility
  }

  bool serializeAllKeysToSingleFile(const std::string& filename) {
    LOG_INFO("Serializing all keys to single file: {}", filename);

    // Create vector of automorphism indices
    std::vector<usint> automorphismIndices;
    for (const auto& rotIndex : rotIndices) {
      automorphismIndices.push_back(getAutomorphismIndex(rotIndex));
    }

    // Open file for writing
    std::ofstream keyFile(filename, std::ios::binary);
    if (!keyFile) {
      LOG_ERROR("Could not open file for writing: {}", filename);
      return false;
    }

    // Serialize all keys at once
    bool success = cc->SerializeEvalAutomorphismKey(
        keyFile, SerType::BINARY, keyTag, automorphismIndices);

    if (success) {
      LOG_INFO("Successfully serialized {} keys to {}",
               automorphismIndices.size(), filename);
    } else {
      LOG_ERROR("Failed to serialize keys to {}", filename);
    }

    return success;
  }

  bool deserializeAllKeysFromSingleFile(const std::string& filename) {
    LOG_INFO("Deserializing all keys from single file: {}", filename);

    // Create vector of automorphism indices
    std::vector<usint> automorphismIndices;
    for (const auto& rotIndex : rotIndices) {
      automorphismIndices.push_back(getAutomorphismIndex(rotIndex));
    }

    // Open file for reading
    std::ifstream keyFile(filename, std::ios::binary);
    if (!keyFile) {
      LOG_ERROR("Could not open file for reading: {}", filename);
      return false;
    }

    // Deserialize all keys at once
    bool success = cc->DeserializeEvalAutomorphismKey(
        keyFile, SerType::BINARY, keyTag, automorphismIndices);

    if (success) {
      LOG_INFO("Successfully deserialized {} keys from {}",
               automorphismIndices.size(), filename);
      // Mark all keys as loaded
      for (const auto& rotIndex : rotIndices) {
        loadedKeys.insert(rotIndex);
      }
    } else {
      LOG_ERROR("Failed to deserialize keys from {}", filename);
    }

    return success;
  }

  bool serializeAllKeys() {
    bool force = BenchmarkCLI::getSerForce();
    bool singleFile = BenchmarkCLI::getSerSingleFile();

    if (force == false && operationMode == KeyMemMode::IGNORE) {
      LOG_DEBUG("IGNORE mode: Skipping serialize all keys");
      return true;
    }

    if (singleFile) {
      return serializeAllKeysToSingleFile("/tmp/rotation_key_all.bin");
    }

    LOG_INFO("Serializing all keys individually ({} rotation indices)",
             rotIndices.size());
    bool success = true;
    for (const auto& rotIndex : rotIndices) {
      success &= serializeKey(rotIndex);
    }

    if (success) {
      LOG_INFO("Successfully serialized all {} keys", rotIndices.size());
    } else {
      LOG_ERROR("Failed to serialize some keys");
    }

    return success;
  }

  bool deserializeAllKeys() {
    bool force = BenchmarkCLI::getSerForce();
    bool singleFile = BenchmarkCLI::getSerSingleFile();

    if (force == false && operationMode == KeyMemMode::IGNORE) {
      LOG_DEBUG("IGNORE mode: Skipping deserialize all keys");
      return true;
    }

    if (singleFile) {
      return deserializeAllKeysFromSingleFile("/tmp/rotation_key_all.bin");
    }

    LOG_INFO("Deserializing all keys individually ({} rotation indices)",
             rotIndices.size());
    bool success = true;
    for (const auto& rotIndex : rotIndices) {
      success &= deserializeKey(rotIndex);
    }

    if (success) {
      LOG_INFO("Successfully deserialized all {} keys", rotIndices.size());
    } else {
      LOG_ERROR("Failed to deserialize some keys");
    }

    return success;
  }

  bool clearAllKeys() {
    bool serForce = BenchmarkCLI::getSerForce();

    if (serForce == false && operationMode == KeyMemMode::IGNORE) {
      LOG_DEBUG("IGNORE mode: Skipping clear all keys");
      return true;
    }

    LOG_INFO("Clearing all keys for tag: {}", keyTag);
    cc->ClearEvalAutomorphismKeys(keyTag);
    auto numCleared = loadedKeys.size();
    loadedKeys.clear();
    LOG_INFO("Successfully cleared {} keys", numCleared);

    return true;
  }

  std::string getKeyFilename(int rotationIndex) const {
    return "/tmp/rotation_key_" + std::to_string(rotationIndex) + ".bin";
  }

  bool checkKeyExists(int rotationIndex) const {
    std::string filename = getKeyFilename(rotationIndex);
    std::ifstream keyFile(filename);
    bool exists = keyFile.good();
    if (exists) {
      LOG_DEBUG("Key file for rotation index {} exists: {}", rotationIndex,
                filename);
    } else {
      LOG_DEBUG("Key file for rotation index {} does not exist: {}",
                rotationIndex, filename);
    }
    return exists;
  }

  // Print stats about key operations
  void printKeyStats() const {
    LOG_INFO("KeyMemRT Stats:");
    LOG_INFO("  Mode: {}", getModeString(operationMode));
    LOG_INFO("  Total rotation indices: {}", rotIndices.size());
    LOG_INFO("  Keys loaded: {}", loadedKeys.size());

    std::cout << "KeyMemRT Stats:\n"
              << "  Mode: " << getModeString(operationMode) << "\n"
              << "  Total rotation indices: " << rotIndices.size() << "\n"
              << "  Keys loaded: " << loadedKeys.size() << "\n";
  }

  // Enable/disable logging to file
  void enableFileLogging(const std::string& filename = "keymemrt.log") {
    Logger::getInstance().setLogToFile(true, filename);
    LOG_INFO("File logging enabled: {}", filename);
  }

  // Set log level
  void setLogLevel(LogLevel level) {
    Logger::getInstance().setLogLevel(level);
    LOG_INFO("Log level set to: {}",
             Logger::getInstance().levelToString(level));
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
