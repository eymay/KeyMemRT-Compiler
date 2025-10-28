// RUN: heir-opt --kmrt-key-prefetching="prefetch-threshold=30" %s | FileCheck %s

!Z536903681_i64 = !mod_arith.int<536903681 : i64>
!Z66813953_i64 = !mod_arith.int<66813953 : i64>
!Z66961409_i64 = !mod_arith.int<66961409 : i64>
!Z66994177_i64 = !mod_arith.int<66994177 : i64>
!Z67043329_i64 = !mod_arith.int<67043329 : i64>
!Z67239937_i64 = !mod_arith.int<67239937 : i64>
!cc = !openfhe.crypto_context
!rk = !kmrt.rot_key<>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 26>
#inverse_canonical_encoding1 = #lwe.inverse_canonical_encoding<scaling_factor = 1>
#key = #lwe.key<>
#modulus_chain_L5_C5 = #lwe.modulus_chain<elements = <536903681 : i64, 67043329 : i64, 66994177 : i64, 67239937 : i64, 66961409 : i64, 66813953 : i64>, current = 5>
#ring_f64_1 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1>>
#ring_f64_1_x8192 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**8192>>
!rns_L5 = !rns.rns<!Z536903681_i64, !Z67043329_i64, !Z66994177_i64, !Z67239937_i64, !Z66961409_i64, !Z66813953_i64>
!pt = !lwe.new_lwe_plaintext<application_data = <message_type = tensor<4xf64>>, plaintext_space = <ring = #ring_f64_1, encoding = #inverse_canonical_encoding1>>
#ring_rns_L5_1_x8192 = #polynomial.ring<coefficientType = !rns_L5, polynomialModulus = <1 + x**8192>>
#ciphertext_space_L5 = #lwe.ciphertext_space<ring = #ring_rns_L5_1_x8192, encryption_type = mix>
!ct_L5 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<4xf64>>, plaintext_space = <ring = #ring_f64_1_x8192, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L5, key = #key, modulus_chain = #modulus_chain_L5_C5>

// Test case for index type handling in key prefetching
// This test uses load_key with an index type directly (not cast to i64)
// The prefetching pass should handle this correctly by casting to i64

// CHECK-LABEL: func.func @test_index_type_load_key
// CHECK-SAME: (%[[CC:.*]]: !cc, %[[CT:.*]]: !ct_L5)
func.func @test_index_type_load_key(%cc: !cc, %ct: !ct_L5) -> !ct_L5 {
  // Before the loop, prefetch the first iteration with proper type
  // CHECK: %[[C1:.*]] = arith.constant 1
  // CHECK-NEXT: kmrt.prefetch_key %[[C1]]

  // CHECK: %[[RESULT:.*]] = affine.for %[[IV:.*]] = 1 to 8 iter_args(%[[ACC:.*]] = %{{.*}}) -> (!ct_L5) {
  %result = affine.for %iv = 1 to 8 iter_args(%acc = %ct) -> (!ct_L5) {
    // Inside the loop, prefetch key for next iteration
    // CHECK: affine.if #{{.*}}(%[[IV]]) {
    // CHECK:   %[[NEXT_IV:.*]] = affine.apply #{{.*}}(%[[IV]])
    // CHECK:   %[[NEXT_IV_I64:.*]] = arith.index_cast %[[NEXT_IV]]
    // CHECK:   kmrt.prefetch_key %[[NEXT_IV_I64]]
    // CHECK: }

    // Load rotation key using index type directly (no explicit cast)
    // CHECK: %[[RK:.*]] = kmrt.load_key %[[IV]]
    %rk = kmrt.load_key %iv : index -> !rk

    // Rotate the ciphertext
    // CHECK: %[[CT_ROT:.*]] = openfhe.rot %[[CC]], %[[ACC]], %[[RK]]
    %ct_rot = openfhe.rot %cc, %acc, %rk : (!cc, !ct_L5, !rk) -> !ct_L5

    // Clear the key after use
    // CHECK: kmrt.clear_key %[[RK]]
    kmrt.clear_key %rk : !rk

    // CHECK: affine.yield %[[CT_ROT]]
    affine.yield %ct_rot : !ct_L5
  }

  // CHECK: return %[[RESULT]]
  return %result : !ct_L5
}
