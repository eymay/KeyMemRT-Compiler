# Rotation Depth Analysis pass
package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

cc_library(
    name = "RotationDepthAnalysis",
    srcs = ["RotationDepthAnalysis.cpp"],
    hdrs = ["RotationDepthAnalysis.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "RotationDepthAnalysis",
    td_file = "RotationDepthAnalysis.td",
)
