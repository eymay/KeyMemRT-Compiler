load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

td_library(
    name = "td_files",
    srcs = ["OpenfheToKMRT.td"],
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "pass_inc_gen",
    tbl_outs = [
        (
            ["-gen-pass-decls"],
            "OpenfheToKMRT.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "OpenfheToKMRT.td",
    deps = [":td_files"],
)

cc_library(
    name = "OpenfheToKMRT",
    srcs = ["OpenfheToKMRT.cpp"],
    hdrs = ["OpenfheToKMRT.h"],
    deps = [
        ":pass_inc_gen",
        "//lib/Dialect/KMRT/IR:Dialect",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect", 
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)
