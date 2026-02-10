# KeyMemRT Compiler

KeyMemRT is an FHE (Fully Homomorphic Encryption) compiler that reduces
the memory footprint of FHE applications. It reuses the IR and infrastructure
of [HEIR](https://github.com/google/heir).

More information is available in the KeyMemRT paper [here](https://arxiv.org/abs/2601.18445).

## Getting Started

KeyMemRT Compiler is built the same way as HEIR. Please follow HEIR
[docs](https://heir.dev/docs/getting_started/#building-from-source) to build from source.

Alternatively, here are the commands for an Ubuntu/x86 machine:

1. Install the prerequisites:
```shell
sudo apt update && sudo apt install wget python3 \
clang lld libomp-dev \
mold zlib1g-dev
```
2. Install `Bazelisk`
```shell
wget -c https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 && \
mv bazelisk-linux-amd64 bazel && \
chmod +x bazel && \
mkdir -p ~/bin && \
echo 'export PATH=$PATH:~/bin' >> ~/.bashrc && \
mv bazel ~/bin/bazel && \
source ~/.bashrc
```
3. Clone and build the repo:
```shell
git clone https://github.com/eymay/KeyMemRT-Compiler.git && cd KeyMemRT-Compiler/
bazel build @heir//tools:heir-opt
bazel build @heir//tools:heir-translate
```
KeyMemRT Compiler is a series of passes that can be called from the newly built driver `./bazel-bin/tools/heir-opt`.

4. Run KeyMemRT tests:
```shell
bazel test //tests:keymemrt_tests
```

## KeyMemRT Pipeline

KeyMemRT adds the passes listed below:
```shell
--lower-linear-transform        
--symbolic-bsgs-decomposition   
--kmrt-merge-rotation-keys      
--kmrt-key-prefetching          
--profile-annotator
--remove-unnecessary-bootstraps
--bootstrap-rotation-analysis
--openfhe-insert-clear-ops      
--kmrt-key-prefetching
```
KeyMemRT reuses these HEIR and MLIR passes, some with modifications:
```shell
--ckks-to-lwe
--lwe-to-openfhe
--openfhe-configure-crypto-context  
--openfhe-fast-rotation-precompute
--cse
--lower-affine
```

An example pipeline to lower the HEIR `ckks` dialect to `openfhe` dialect with KeyMemRT key management is:
```
./bazel-bin/tools/heir-opt
	--ckks-to-lwe \ 
    --lwe-to-openfhe \
	--lower-linear-transform \
	--symbolic-bsgs-decomposition \
    --kmrt-merge-rotation-keys \
	--annotate-module="backend=openfhe scheme=ckks" \
	--openfhe-configure-crypto-context \
    --openfhe-fast-rotation-precompute \
    --bootstrap-rotation-analysis \
    --kmrt-merge-rotation-keys \
    --cse \
    --openfhe-insert-clear-ops \
    <ckks-input.mlir> > <static-out.mlir>
```
This will result in the KeyMemRT Low Memory mode execution that disk IO and computation will block each other.
To enable prefetching and to get to KeyMemRT Balanced mode, these passes can be added in the end:

```
...
--kmrt-key-prefetching="runtime-delegated=1" \
--lower-affine` \
```

## Cite

```bibtex
@misc{ünay2026keymemrt,
      title={KeyMemRT Compiler and Runtime: Unlocking Memory-Scalable FHE}, 
      author={Eymen Ünay and Björn Franke and Jackson Woodruff},
      year={2026},
      eprint={2601.18445},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2601.18445}, 
}
```
