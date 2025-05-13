# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "halevi_shoup_matmul_test",
    generated_lib_header = "halevi_shoup_matmul_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=16",
        "--scheme-to-openfhe",
    ] + ["--merge-eval-keys",
         "--rotation-decompose=base-set-size=500 max-chain-length=2",
	 "--cse",
         "--merge-eval-keys"],
    heir_translate_flags = [
        "--openfhe-include-type=source-relative",
    ],
    mlir_src = "@heir//tests/Examples/common:halevi_shoup_matmul.mlir",
    tags = ["notap"],
    test_src = "halevi_shoup_matmul_test.cpp",
)
