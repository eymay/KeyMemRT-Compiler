load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "Transforms",
    hdrs = [
        "Passes.h",
        "WrapRotations.h",
    ],
    deps = [
        "//lib/Dialect/KMRT/IR:Dialect",
    ],
)
