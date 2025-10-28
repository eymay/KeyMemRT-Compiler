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

// Test with affine loop containing rotations (like a linear transform)
// Prefetch should:
// 1. Insert prefetches for first N iterations before the loop
// 2. Inside the loop, prefetch key for iteration (i + N) wrapped in affine.if to avoid overflow

// CHECK-LABEL: func.func @affine_loop_linear_transform
// CHECK-SAME: (%[[CC:.*]]: !cc, %[[CT:.*]]: !ct_L5)
func.func @affine_loop_linear_transform(%cc: !cc, %ct: !ct_L5) -> !ct_L5 {
  %cst = arith.constant dense<[
    [1.0, 1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0, 2.0],
    [3.0, 3.0, 3.0, 3.0],
    [4.0, 4.0, 4.0, 4.0],
    [5.0, 5.0, 5.0, 5.0],
    [6.0, 6.0, 6.0, 6.0],
    [7.0, 7.0, 7.0, 7.0],
    [8.0, 8.0, 8.0, 8.0]
  ]> : tensor<8x4xf64>

  // Initialize accumulator
  %extracted_slice_0 = tensor.extract_slice %cst[0, 0] [1, 4] [1, 1] : tensor<8x4xf64> to tensor<4xf64>
  %pt_0 = lwe.rlwe_encode %extracted_slice_0 {encoding = #inverse_canonical_encoding1, ring = #ring_f64_1} : tensor<4xf64> -> !pt
  %ct_init = openfhe.mul_plain %cc, %ct, %pt_0 : (!cc, !ct_L5, !pt) -> !ct_L5

  // Before the loop, prefetch the first iteration(s) based on cost analysis
  // Loop body cost: rot(15) + mul_plain(10) + add(1) ≈ 26
  // With threshold=30, prefetch distance = 30/26 ≈ 1 iteration ahead
  // So we prefetch only iteration 1 before the loop
  // CHECK: %[[C1:.*]] = arith.constant 1
  // CHECK-NEXT: kmrt.prefetch_key %[[C1]]

  // CHECK: %[[RESULT:.*]] = affine.for %[[IV:.*]] = 1 to 8 iter_args(%[[ACC:.*]] = %{{.*}}) -> (!ct_L5) {
  %result = affine.for %iv = 1 to 8 iter_args(%acc = %ct_init) -> (!ct_L5) {
    // Inside the loop, prefetch key for next iteration (i + 1)
    // This is wrapped in affine.if to avoid prefetching beyond loop bounds
    // The condition is: i + 1 < 8, which is i < 7, which is i <= 7 in affine form
    // So iterations 1-7 will prefetch keys 2-8 respectively
    // CHECK: affine.if #{{.*}}(%[[IV]]) {
    // CHECK:   %[[NEXT_IV:.*]] = affine.apply #{{.*}}(%[[IV]])
    // CHECK:   %[[NEXT_IV_I64:.*]] = arith.index_cast %[[NEXT_IV]]
    // CHECK:   kmrt.prefetch_key %[[NEXT_IV_I64]]
    // CHECK: }

    // Load rotation key for current iteration
    // CHECK: %[[IV_I64:.*]] = arith.index_cast %[[IV]]
    %iv_i64 = arith.index_cast %iv : index to i64
    // CHECK: %[[RK:.*]] = kmrt.load_key %[[IV_I64]]
    %rk = kmrt.load_key %iv_i64 : i64 -> !rk

    // Rotate the ciphertext
    // CHECK: %[[CT_ROT:.*]] = openfhe.rot %[[CC]], %[[CT]], %[[RK]]
    %ct_rot = openfhe.rot %cc, %ct, %rk : (!cc, !ct_L5, !rk) -> !ct_L5

    // Clear the key after use
    // CHECK: kmrt.clear_key %[[RK]]
    kmrt.clear_key %rk : !rk

    // Extract slice and multiply
    %extracted_slice = tensor.extract_slice %cst[%iv, 0] [1, 4] [1, 1] : tensor<8x4xf64> to tensor<4xf64>
    %pt = lwe.rlwe_encode %extracted_slice {encoding = #inverse_canonical_encoding1, ring = #ring_f64_1} : tensor<4xf64> -> !pt
    // CHECK: %[[CT_MUL:.*]] = openfhe.mul_plain
    %ct_mul = openfhe.mul_plain %cc, %ct_rot, %pt : (!cc, !ct_L5, !pt) -> !ct_L5

    // Accumulate
    // CHECK: %[[CT_ADD:.*]] = openfhe.add
    %ct_add = openfhe.add %cc, %acc, %ct_mul : (!cc, !ct_L5, !ct_L5) -> !ct_L5

    // CHECK: affine.yield %[[CT_ADD]]
    affine.yield %ct_add : !ct_L5
  }

  // CHECK: return %[[RESULT]]
  return %result : !ct_L5
}
