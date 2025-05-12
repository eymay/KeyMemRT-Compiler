#include <fstream>
#include <iostream>
#include <vector>

#include "KeyMemRT.hpp"
#include "ResourceMonitor.hpp"

KeyMemRT keymem_rt;

#include "mlp_openfhe_keymemrt.h"

#define DIM 1024

struct Sample {
  int label;
  float image[DIM];
};

using Dataset = std::vector<Sample>;

void load_dataset(Dataset &dataset, const char *filename) {
  std::ifstream file(filename);
  Sample sample;
  while (file >> sample.label) {
    for (int i = 0; i < DIM; i++) {
      file >> sample.image[i];
    }
    dataset.push_back(sample);
  }
}

template <int N>
int argmax(float *A) {
  int max_idx = 0;
  for (int i = 1; i < N; i++) {
    if (A[i] > A[max_idx]) {
      max_idx = i;
    }
  }
  return max_idx;
}

int main(int argc, char *argv[]) {
  // Initialize KeyMemRT from command line args
  keymem_rt.initFromArgs(argc, argv);

  auto dataset = Dataset();
  load_dataset(dataset, "mnist_test.txt");
  int accurate = 0;
  int total = 10;  // 10000

  ResourceMonitor monitor;
  monitor.start();

  // Mark the start of the entire experiment
  monitor.mark_event_start("experiment");

  // Mark crypto context initialization
  monitor.mark_event_start("crypto_init");
  auto cryptoContext = mlp__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;

  cryptoContext = mlp__configure_crypto_context(cryptoContext, secretKey);
  std::cout << *cryptoContext->GetCryptoParameters() << std::endl;
  monitor.mark_event_end("crypto_init");

  for (int i = 0; i < total; ++i) {
    // Create a unique event name for this sample
    std::string sample_event = "sample_" + std::to_string(i);
    monitor.mark_event_start(sample_event);

    auto *input = dataset[i].image;
    std::vector<float> input_vector(input, input + DIM);

    // Mark encryption phase
    monitor.mark_event_start("encrypt_" + std::to_string(i));
    auto input_encrypted =
        mlp__encrypt__arg0(cryptoContext, input_vector, publicKey);
    monitor.mark_event_end("encrypt_" + std::to_string(i));

    // Mark MLP inference run
    monitor.mark_event_start("mlp_run_" + std::to_string(i));
    auto output_encrypted = mlp(cryptoContext, input_encrypted);
    monitor.mark_event_end("mlp_run_" + std::to_string(i));

    // Mark decryption phase
    monitor.mark_event_start("decrypt_" + std::to_string(i));
    std::vector<float> output =
        mlp__decrypt__result0(cryptoContext, output_encrypted, secretKey);
    monitor.mark_event_end("decrypt_" + std::to_string(i));

    auto max_id = argmax<1024>(output.data());
    auto label = dataset[i].label;
    std::cout << "max_id: " << max_id << ", label: " << label << std::endl;
    if (max_id == label) {
      accurate++;
    }

    // End the sample event
    monitor.mark_event_end(sample_event);
  }

  // Mark the end of the experiment
  monitor.mark_event_end("experiment");

  // Stop monitoring and save results
  monitor.stop();
  std::string filename = BenchmarkCLI::getOutputFilename("mlp_1024") + ".csv";
  monitor.save_to_file(filename);

  std::cout << "accuracy: " << accurate << "/" << total << std::endl;
  keymem_rt.printKeyStats();
  return 0;
}
