#ifndef LIB_TRANSFORMS_MEMORYESTIMATION_MEMORYESTIMATION_TD_
#define LIB_TRANSFORMS_MEMORYESTIMATION_MEMORYESTIMATION_TD_

include "mlir/Pass/PassBase.td"

def MemoryEstimation : Pass<"memory-estimation"> {
  let summary = "Estimate memory usage using unified liveness analysis";
  let description = [{
    This pass uses the unified liveness analysis to estimate actual memory usage 
    in bytes for FHE programs. It converts tower numbers into concrete memory 
    estimates using the mathematical formulas for RNS representation:

    For keys: 2×64Ndnum(k+k′)/8 = 16Ndnum(k+k′) bytes
    - Where k = Q towers, k' = P towers, N = ring dimension, dnum = 2 for RLWE

    For ciphertexts: 2×64Ndk/8 = 16Ndk bytes  
    - Where k = Q towers only (no P towers), N = ring dimension, dnum = 2

    The pass provides multiple output formats:
    1. Text format: Human-readable report with summary statistics and memory distribution
    2. Matrix format: Visual matrix showing live values (*=ciphertext, +=key) across operations
    3. JSON format: Structured data suitable for visualization tools and further analysis

    Key features:
    - Peak memory usage identification
    - Memory hotspot detection
    - Per-operation memory breakdown
    - Memory distribution analysis
    - Byte-accurate memory calculations

    This pass depends on the unified liveness analysis and is intended for:
    - Memory optimization planning
    - Performance analysis
    - Resource allocation decisions
    - Memory-constrained deployment planning
  }];

  let options = [
    Option<"outputFormat", "output-format", "std::string",
           /*default=*/"\"text\"", 
           "Output format: 'text', 'matrix', or 'json'">,
    Option<"outputFile", "output-file", "std::string",
           /*default=*/"\"\"", 
           "Output file path (empty means stderr)">,
    Option<"ringDim", "ring-dim", "unsigned",
           /*default=*/"4096", 
           "Ring dimension N for memory calculations">,
    Option<"dnum", "dnum", "unsigned",
           /*default=*/"2", 
           "Dnum parameter (typically 2 for RLWE)">,
    Option<"fixedPTowers", "fixed-p-towers", "unsigned",
           /*default=*/"4", 
           "Fixed number of P towers for keys">,
    Option<"showHotspots", "show-hotspots", "bool",
           /*default=*/"true", 
           "Include memory hotspot identification">,
    Option<"memoryThresholdMB", "memory-threshold-mb", "unsigned",
           /*default=*/"1024", 
           "Memory threshold in MB for hotspot identification">,
    Option<"opScale", "op-scale", "bool",
           /*default=*/"true", 
           "Output format: 'text', 'matrix', or 'json'">,
  ];
  
  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];
}

#endif  // LIB_TRANSFORMS_MEMORYESTIMATION_MEMORYESTIMATION_TD_
