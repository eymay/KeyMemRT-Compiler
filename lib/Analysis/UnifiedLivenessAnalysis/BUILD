# RotationAnalysis analysis pass
package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "UnifiedLivenessAnalysis",
    srcs = ["UnifiedLivenessAnalysis.cpp"],
    hdrs = ["UnifiedLivenessAnalysis.h"],
    deps = [
        "@heir//lib/Dialect/KMRT/IR:Dialect",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:ArithDialect",
    ],
)
