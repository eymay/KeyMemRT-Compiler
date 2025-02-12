# RotationAnalysis analysis pass
package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "RotationKeyLivenessAnalysis",
    srcs = ["RotationKeyLivenessAnalysis.cpp"],
    hdrs = ["RotationKeyLivenessAnalysis.h"],
    deps = [
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@heir//lib/Dialect/TensorExt/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
    ],
)
