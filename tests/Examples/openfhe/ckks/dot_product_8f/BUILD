# See README.md for setup required to run these tests

load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "dot_product_8f_test",
    generated_lib_header = "dot_product_8f_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=8",
        "--scheme-to-openfhe",
    ] + [#"--merge-eval-keys",
         # "--rotation-decompose=base-set-size=500 max-chain-length=2",
	 "--cse",
         # "--merge-eval-keys",
         "--key-compression"],
    heir_translate_flags = [],
    mlir_src = "@heir//tests/Examples/common:dot_product_8f.mlir",
    tags = ["notap"],
    test_src = "dot_product_8f_test.cpp",
)
