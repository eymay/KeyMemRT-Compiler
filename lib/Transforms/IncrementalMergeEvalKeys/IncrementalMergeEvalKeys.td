#ifndef LIB_TRANSFORMS_INCREMENTALMERGEEVALKEYS_INCREMENTALMERGEEVALKEYS_TD_
#define LIB_TRANSFORMS_INCREMENTALMERGEEVALKEYS_INCREMENTALMERGEEVALKEYS_TD_

include "mlir/Pass/PassBase.td"


def IncrementalMergeEvalKeys : Pass<"incremental-merge-eval-keys"> {
  let summary = "Incremental merge evaluation keys with work range tracking";
  let description = [{
    Uses incremental liveness analysis to optimize when rotation keys are 
    generated and freed. Only re-analyzes regions that changed in previous 
    iterations, providing significant performance improvements for large codebases.
    
    This is an optimized version of merge-eval-keys that tracks work ranges
    and uses analysis caching to avoid redundant dataflow analysis.
  }];
  
  let options = [
    Option<"maxIterations", "max-iterations", "unsigned", /*default=*/"10",
           "Maximum number of iterations before stopping">,
    Option<"maxMergeDistance", "max-merge-distance", "unsigned", /*default=*/"25", 
           "Maximum distance for merging deserialize/clear pairs">,
    Option<"enableDebug", "enable-debug", "bool", /*default=*/"false",
           "Enable debug output for incremental analysis">
  ];
}


#endif  // LIB_TRANSFORMS_INCREMENTALMERGEEVALKEYS_INCREMENTALMERGEEVALKEYS_TD_
