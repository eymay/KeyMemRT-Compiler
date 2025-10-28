load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "BootstrapRotationAnalysis",
    srcs = ["BootstrapRotationAnalysis.cpp"],
    hdrs = ["BootstrapRotationAnalysis.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/KMRT/IR:Dialect",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//llvm:Support",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "BootstrapRotationAnalysis",
    td_file = "BootstrapRotationAnalysis.td",
)
