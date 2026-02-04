load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = [
        "Passes.h",
    ],
    deps = [
        ":KeyPrefetching",
        ":MergeRotationKeys",
        ":pass_inc_gen",
        "//lib/Dialect/KMRT/IR:Dialect",
    ],
)

cc_library(
    name = "KeyPrefetching",
    srcs = ["KeyPrefetching.cpp"],
    hdrs = [
        "KeyPrefetching.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/KMRT/IR:Dialect",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "MergeRotationKeys",
    srcs = ["MergeRotationKeys.cpp"],
    hdrs = [
        "MergeRotationKeys.h",
    ],
    deps = [
        ":pass_inc_gen",
        "@heir//lib/Dialect/KMRT/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = {
        "Passes.h.inc": [
            "-gen-pass-decls",
            "-name=KMRT",
        ],
        "KMRTPasses.md": ["-gen-pass-doc"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Passes.td",
    td_srcs = [
        "KeyPrefetching.td",
        "MergeRotationKeys.td",
    ],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)
