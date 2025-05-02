package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "OptimizeRotationBaseAnalysis",
    srcs = ["OptimizeRotationBaseAnalysis.cpp"],
    hdrs = ["OptimizeRotationBaseAnalysis.h"],
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@com_google_ortools//ortools/math_opt/cpp:math_opt",
        "@com_google_ortools//ortools/math_opt/solvers:gscip_solver",
        "@heir//lib/Dialect/Openfhe/IR:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
