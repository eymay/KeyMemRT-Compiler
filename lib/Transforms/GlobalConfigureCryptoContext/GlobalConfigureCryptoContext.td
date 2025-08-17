#ifndef LIB_TRANSFORMS_GLOBALCONFIGURECRYPTOCONTEXT_GLOBALCONFIGURECRYPTOCONTEXT_TD_
#define LIB_TRANSFORMS_GLOBALCONFIGURECRYPTOCONTEXT_GLOBALCONFIGURECRYPTOCONTEXT_TD_

include "mlir/Pass/PassBase.td"

def GlobalConfigureCryptoContext : Pass<"global-configure-crypto-context"> {
  // FIXME: add add summary/description
 let summary = "Link and merge crypto context configurations from multiple MLIR files";
  let description = [{
    This pass operates on modules that contain multiple configure crypto context functions
    from different MLIR files that have been combined. It analyzes all individual configure
    functions, extracts their configurations, and creates a single global configure function
    that represents the union of all features and the maximum of all performance parameters.
    
    Features are ORed together:
    - If any file needs relinearization keys, the global config will include them
    - If any file needs bootstrap, the global config will include it
    - Rotation key sets are united (all unique rotation indices are included)
    
    Performance parameters are maximized:
    - Multiplicative depth is set to the maximum across all files
    - Ring dimension and other crypto parameters must be consistent across files
    
    The pass removes individual configure functions and replaces them with a single
    global__configure_crypto_context function.
  }];
  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];
}

#endif  // LIB_TRANSFORMS_GLOBALCONFIGURECRYPTOCONTEXT_GLOBALCONFIGURECRYPTOCONTEXT_TD_
