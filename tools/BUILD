# KeyMemRT tools

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_binary(
    name = "keymemrt-opt",
    srcs = ["keymemrt-opt.cpp"],
    includes = ["include"],
    deps = [
        # Core HEIR interfaces
        "@heir//lib/Dialect:HEIRInterfaces",
        # KeyMemRT dialects
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@heir//lib/Dialect/CKKS/Conversions/CKKSToLWE",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/LWE/Conversions/LWEToOpenfhe",
        "@heir//lib/Dialect/LWE/Transforms",
        "@heir//lib/Dialect/KMRT/IR:Dialect",
        "@heir//lib/Dialect/KMRT/Transforms",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@heir//lib/Dialect/Openfhe/Transforms",
        "@heir//lib/Dialect/Openfhe/Transforms:ConfigureCryptoContext",
        "@heir//lib/Dialect/Orion/IR:Dialect",
        # Supporting dialects
        "@heir//lib/Dialect/ModArith/IR:Dialect",
        "@heir//lib/Dialect/Polynomial/IR:Dialect",
        "@heir//lib/Dialect/RNS/IR:Dialect",
        "@heir//lib/Dialect/RNS/IR:RNSTypeInterfaces",
        "@heir//lib/Dialect/Random/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        # KeyMemRT transforms
        "@heir//lib/Transforms/AddRotationKeys",
        "@heir//lib/Transforms/AnnotateModule",
        "@heir//lib/Transforms/BootstrapRotationAnalysis",
        "@heir//lib/Transforms/LowerLinearTransform",
        "@heir//lib/Transforms/SymbolicBSGSDecomposition",
        "@heir//lib/Transforms/ProfileAnnotator",
        "@heir//lib/Transforms/UnnecessaryBootstrapRemoval",
        # MLIR core
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineToStandard",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_binary(
    name = "keymemrt-translate",
    srcs = ["keymemrt-translate.cpp"],
    includes = ["include"],
    deps = [
        "@heir//lib/Target/Metadata:MetadataEmitter",
        "@heir//lib/Target/OpenFhePke:OpenFheRegistration",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)
