#ifndef LIB_TRANSFORMS_KEYCOMPRESSION_KEYCOMPRESSION_TD_
#define LIB_TRANSFORMS_KEYCOMPRESSION_KEYCOMPRESSION_TD_

include "mlir/Pass/PassBase.td"

def KeyCompression : Pass<"key-compression"> {
  let summary = "Analyze key usage and add compression information using result_towers attributes";
  let description = [{
    This pass analyzes the multiplication depth at which rotation keys are used
    and adds compression information to key clear operations. This allows for
    loading the keys at the appropriate compression level in the future, which
    can reduce memory usage.
    
    This pass expects operations to already be annotated with the result_towers
    attribute by the profile-annotator pass. It uses these annotations to
    determine the appropriate compression levels for rotation keys.
  }];
}

#endif  // LIB_TRANSFORMS_KEYCOMPRESSION_KEYCOMPRESSION_TD_
