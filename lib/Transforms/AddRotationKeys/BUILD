load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "AddRotationKeys",
    srcs = ["AddRotationKeys.cpp"],
    hdrs = ["AddRotationKeys.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "AddRotationKeys",
    td_file = "AddRotationKeys.td",
)
