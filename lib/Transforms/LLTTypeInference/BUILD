load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "LLTTypeInference",
    srcs = ["LLTTypeInference.cpp"],
    hdrs = ["LLTTypeInference.h"],
    deps = [
        ":pass_inc_gen",
        "@llvm-project//mlir:FuncDialect",
        "@heir//lib/Dialect/LWE/IR:Dialect",
        "@heir//lib/Dialect/CKKS/IR:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "LLTTypeInference",
    td_file = "LLTTypeInference.td",
)
