load("@heir//lib/Transforms:transforms.bzl", "add_heir_transforms")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "LoopTiling",
    srcs = ["LoopTiling.cpp"],
    hdrs = ["LoopTiling.h"],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AffineAnalysis",
	  "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

add_heir_transforms(
    generated_target_name = "pass_inc_gen",
    pass_name = "LoopTiling",
    td_file = "LoopTiling.td",
)
