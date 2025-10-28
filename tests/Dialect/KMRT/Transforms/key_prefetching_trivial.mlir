// RUN: heir-opt --kmrt-key-prefetching %s | FileCheck %s

!Z536903681_i64 = !mod_arith.int<536903681 : i64>
!Z66813953_i64 = !mod_arith.int<66813953 : i64>
!Z66961409_i64 = !mod_arith.int<66961409 : i64>
!Z66994177_i64 = !mod_arith.int<66994177 : i64>
!Z67043329_i64 = !mod_arith.int<67043329 : i64>
!Z67239937_i64 = !mod_arith.int<67239937 : i64>
!cc = !openfhe.crypto_context
!rk5 = !kmrt.rot_key<rotation_index = 5>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 26>
#key = #lwe.key<>
#modulus_chain_L5_C5 = #lwe.modulus_chain<elements = <536903681 : i64, 67043329 : i64, 66994177 : i64, 67239937 : i64, 66961409 : i64, 66813953 : i64>, current = 5>
#ring_f64_1_x8192 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**8192>>
!rns_L5 = !rns.rns<!Z536903681_i64, !Z67043329_i64, !Z66994177_i64, !Z67239937_i64, !Z66961409_i64, !Z66813953_i64>
#ring_rns_L5_1_x8192 = #polynomial.ring<coefficientType = !rns_L5, polynomialModulus = <1 + x**8192>>
#ciphertext_space_L5 = #lwe.ciphertext_space<ring = #ring_rns_L5_1_x8192, encryption_type = mix>
!ct_L5 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<4xf64>>, plaintext_space = <ring = #ring_f64_1_x8192, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L5, key = #key, modulus_chain = #modulus_chain_L5_C5>

// Trivial test: A sequence of arithmetic operations followed by a rotation
// The prefetch_key should be inserted early based on operation cost analysis

// CHECK-LABEL: func.func @trivial_sequence
// CHECK-SAME: (%[[CC:.*]]: !cc, %[[CT1:.*]]: !ct_L5, %[[CT2:.*]]: !ct_L5)
func.func @trivial_sequence(%cc: !cc, %ct1: !ct_L5, %ct2: !ct_L5) -> !ct_L5 {
  // Long sequence of operations (cost accumulation)
  // CHECK: %[[C5:.*]] = arith.constant 5
  %c5 = arith.constant 5 : i64

  // Cost = 1 (add)
  // CHECK: %[[V1:.*]] = openfhe.add
  %v1 = openfhe.add %cc, %ct1, %ct2 : (!cc, !ct_L5, !ct_L5) -> !ct_L5

  // Cost = 1+1 = 2 (add)
  // CHECK: %[[V2:.*]] = openfhe.add
  %v2 = openfhe.add %cc, %v1, %ct2 : (!cc, !ct_L5, !ct_L5) -> !ct_L5

  // The prefetch should be inserted after walking back and accumulating enough cost
  // Prefetch is inserted after the second add (cost accumulated going backward)
  // CHECK: kmrt.prefetch_key %[[C5]]

  // Cost = 2+10 = 12 (mul)
  // CHECK: %[[V3:.*]] = openfhe.mul
  %v3 = openfhe.mul %cc, %v2, %ct1 : (!cc, !ct_L5, !ct_L5) -> !ct_L5

  // Cost = 12+1 = 13 (add)
  // CHECK: %[[V4:.*]] = openfhe.add
  %v4 = openfhe.add %cc, %v3, %ct2 : (!cc, !ct_L5, !ct_L5) -> !ct_L5

  // Cost = 13+10 = 23 (mul)
  // CHECK: %[[V5:.*]] = openfhe.mul
  %v5 = openfhe.mul %cc, %v4, %ct1 : (!cc, !ct_L5, !ct_L5) -> !ct_L5

  // Cost = 23+1 = 24 (add)
  // CHECK: %[[V6:.*]] = openfhe.add
  %v6 = openfhe.add %cc, %v5, %ct2 : (!cc, !ct_L5, !ct_L5) -> !ct_L5

  // Cost = 24+10 = 34 (mul)
  // CHECK: %[[V7:.*]] = openfhe.mul
  %v7 = openfhe.mul %cc, %v6, %ct1 : (!cc, !ct_L5, !ct_L5) -> !ct_L5

  // Cost = 34+1 = 35 (add)
  // CHECK: %[[V8:.*]] = openfhe.add
  %v8 = openfhe.add %cc, %v7, %ct2 : (!cc, !ct_L5, !ct_L5) -> !ct_L5

  // Cost = 35+10 = 45 (mul)
  // CHECK: %[[V9:.*]] = openfhe.mul
  %v9 = openfhe.mul %cc, %v8, %ct1 : (!cc, !ct_L5, !ct_L5) -> !ct_L5

  // Cost = 45+1 = 46 (add)
  // CHECK: %[[V10:.*]] = openfhe.add
  %v10 = openfhe.add %cc, %v9, %ct2 : (!cc, !ct_L5, !ct_L5) -> !ct_L5

  // Cost = 46+10 = 56 (mul) - should exceed default threshold of 50
  // CHECK: %[[V11:.*]] = openfhe.mul
  %v11 = openfhe.mul %cc, %v10, %ct1 : (!cc, !ct_L5, !ct_L5) -> !ct_L5

  // Now load the key (should not have prefetch before it since we already inserted one)
  // CHECK: %[[RK:.*]] = kmrt.load_key %[[C5]]
  %rk = kmrt.load_key %c5 : i64 -> !rk5

  // Use the key for rotation
  // CHECK: %[[RESULT:.*]] = openfhe.rot %[[CC]], %[[V11]], %[[RK]]
  %result = openfhe.rot %cc, %v11, %rk : (!cc, !ct_L5, !rk5) -> !ct_L5

  // Clear the key
  // CHECK: kmrt.clear_key %[[RK]]
  kmrt.clear_key %rk : !rk5

  // CHECK: return %[[RESULT]]
  return %result : !ct_L5
}
