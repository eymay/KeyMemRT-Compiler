# Tests for HEIR
load("//bazel:lit.bzl", "glob_lit_tests")

package(
    default_applicable_licenses = ["@heir//:license"],
    default_visibility = ["//visibility:public"],
)

# Bundle together all of the test utilities that are used by tests.
filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":lit.cfg.py",
        "@heir//tools:heir-opt",
        "@heir//tools:heir-translate",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:count",
        "@llvm-project//llvm:lit",
        "@llvm-project//llvm:not",
        "@llvm-project//mlir:libmlir_c_runner_utils.so",
        "@llvm-project//mlir:libmlir_runner_utils.so",
        "@llvm-project//mlir:mlir-runner",
        # copybara: run_lit.sh
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_utilities"],
    # Note: the driver argument is silently ignored for the OSS version of this
    # macro, replacing it with a standard invocation of lit whose config is in
    # lit.cfg.py
    driver = "@heir//tests:run_lit.sh",
    exclude = [
        "Examples/common/*",
        # Has a custom driver
        "Examples/plaintext/*",
    ],
    test_file_exts = ["mlir"],
)

exports_files(
    [
        "run_lit.sh",
    ],
)

test_suite(
    name = "keymemrt_tests",
    tests = [
        # Lower linear transform tests
        "//tests/Transforms/lower_linear_transform:test_linear_transform.mlir.test",
        "//tests/Transforms/symbolic_bsgs_decomposition:test_symbolic_bsgs_decomposition.mlir.test",

        # KMRT tests
        "//tests/Dialect/KMRT/Transforms:merge_rotation_keys_basic.mlir.test",
        "//tests/Dialect/KMRT/Transforms:merge_rotation_keys_bsgs.mlir.test",
        "//tests/Dialect/KMRT/Transforms:merge_rotation_keys_preloop.mlir.test",
        "//tests/Dialect/KMRT/Transforms:merge_rotation_keys_postloop.mlir.test",
        "//tests/Dialect/KMRT/Transforms:merge_rotation_keys_prepost.mlir.test",
        "//tests/Dialect/KMRT/Transforms:merge_rotation_keys_bootstrap.mlir.test",
        "//tests/Dialect/KMRT/Transforms:merge_rotation_keys_two_bootstraps.mlir.test",
        # "//tests/Dialect/KMRT/Transforms:merge_rotation_keys_bootstrap_loop_bootstrap.mlir.test",
        "//tests/Dialect/KMRT/Transforms:configure_crypto_context_bsgs.mlir.test",
        "//tests/Dialect/KMRT/Transforms:key_prefetching_trivial.mlir.test",
        "//tests/Dialect/KMRT/Transforms:key_prefetching_affine.mlir.test",
        "//tests/Dialect/KMRT/Transforms:key_prefetching_bsgs.mlir.test",
        "//tests/Dialect/KMRT/Transforms:key_prefetching_index_type.mlir.test",
        "//tests/Dialect/KMRT/Transforms:key_prefetching_constant.mlir.test",

        # KMRT emitter tests
        "//tests/Dialect/KMRT/Emitters:emit_kmrt_rotation_keys.mlir.test",

        # CKKS tests
        # "//tests/Dialect/CKKS:keymemrt_encoding.mlir.test",
        
        # OpenFHE tests
        # "//tests/Transforms/mlir_to_openfhe_ckks:keymemrt_integration.mlir.test",
        
        "//tests/Dialect/Openfhe/Transforms:fast_rotation_precompute_affine.mlir.test",

        # Add any test from anywhere in the repo
    ],
)
