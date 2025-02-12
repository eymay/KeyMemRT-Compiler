#ifndef KEY_MANAGER_H_
#define KEY_MANAGER_H_

#include <fstream>
#include <memory>
#include <string>

#include "src/pke/include/cryptocontext.h"
#include "src/pke/include/openfhe.h"

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
}

class KeyMemRT {
 public:
  KeyMemRT(CryptoContext<DCRTPoly> context,
           KeyMemMode mode = KeyMemMode::IMPERATIVE)
      : cc(std::move(context)), keyTag(""), operationMode(mode) {}

  KeyMemRT() : cc(nullptr), keyTag(""), operationMode(KeyMemMode::IMPERATIVE) {}

  void setCryptoContext(CryptoContext<DCRTPoly>& context) { cc = context; }

  void setKeyTag(const std::string& tag) { keyTag = tag; }

  void setKeyMemMode(KeyMemMode mode) { operationMode = mode; }

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
    return cc->DeserializeEvalAutomorphismKey(keyFile, SerType::BINARY, keyTag,
                                              {automorphismIndex});
  }

  bool clearKey(int rotationIndex) {
    if (operationMode == KeyMemMode::IGNORE) {
      return true;
    }

    auto automorphismIndex = getAutomorphismIndex(rotationIndex);
    auto keyMap = cc->GetEvalAutomorphismKeyMap(keyTag);
    keyMap.erase(automorphismIndex);
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
    return true;
  }

  std::string getKeyFilename(int rotationIndex) const {
    return "rotation_key_" + std::to_string(rotationIndex) + ".bin";
  }

  bool checkKeyExists(int rotationIndex) const {
    std::ifstream keyFile(getKeyFilename(rotationIndex));
    return keyFile.good();
  }

 private:
  usint getAutomorphismIndex(usint rotationIndex) const {
    return cc->FindAutomorphismIndex(rotationIndex);
  }

  CryptoContext<DCRTPoly> cc;
  std::string keyTag;
  std::vector<int32_t> rotIndices;
  KeyMemMode operationMode;
};
#endif  // KEY_MANAGER_H_
