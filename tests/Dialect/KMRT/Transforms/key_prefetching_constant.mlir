// RUN: heir-opt --kmrt-key-prefetching="prefetch-threshold=50" %s | FileCheck %s

!Z536903681_i64 = !mod_arith.int<536903681 : i64>
!Z66813953_i64 = !mod_arith.int<66813953 : i64>
!Z66961409_i64 = !mod_arith.int<66961409 : i64>
!Z66994177_i64 = !mod_arith.int<66994177 : i64>
!Z67043329_i64 = !mod_arith.int<67043329 : i64>
!Z67239937_i64 = !mod_arith.int<67239937 : i64>
!cc = !openfhe.crypto_context
!rk = !kmrt.rot_key<rotation_index = 5>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 26>
#key = #lwe.key<>
#modulus_chain_L5_C5 = #lwe.modulus_chain<elements = <536903681 : i64, 67043329 : i64, 66994177 : i64, 67239937 : i64, 66961409 : i64, 66813953 : i64>, current = 5>
#ring_f64_1_x8192 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**8192>>
!rns_L5 = !rns.rns<!Z536903681_i64, !Z67043329_i64, !Z66994177_i64, !Z67239937_i64, !Z66961409_i64, !Z66813953_i64>
#ring_rns_L5_1_x8192 = #polynomial.ring<coefficientType = !rns_L5, polynomialModulus = <1 + x**8192>>
#ciphertext_space_L5 = #lwe.ciphertext_space<ring = #ring_rns_L5_1_x8192, encryption_type = mix>
!ct_L5 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<4xf64>>, plaintext_space = <ring = #ring_f64_1_x8192, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L5, key = #key, modulus_chain = #modulus_chain_L5_C5>

// Test case for constant index handling in key prefetching
// This test verifies that when a load_key uses a constant, the prefetch_key
// creates a NEW constant at the prefetch location to avoid SSA dominance violations

// CHECK-LABEL: func.func @test_constant_dominance
// CHECK-SAME: (%[[CC:.*]]: !cc, %[[CT:.*]]: !ct_L5)
func.func @test_constant_dominance(%cc: !cc, %ct: !ct_L5) -> !ct_L5 {
  // The prefetch should create a NEW constant at the BEGINNING of the function
  // to avoid SSA dominance violations (since the original %c5 is defined later)
  // CHECK: %[[PREFETCH_C5:.*]] = arith.constant 5
  // CHECK-NEXT: kmrt.prefetch_key %[[PREFETCH_C5]]

  // Some operations to accumulate cost
  // CHECK: %[[V1:.*]] = openfhe.mul
  %v1 = openfhe.mul %cc, %ct, %ct : (!cc, !ct_L5, !ct_L5) -> !ct_L5
  // CHECK: %[[V2:.*]] = openfhe.mul
  %v2 = openfhe.mul %cc, %v1, %ct : (!cc, !ct_L5, !ct_L5) -> !ct_L5
  // CHECK: %[[V3:.*]] = openfhe.mul
  %v3 = openfhe.mul %cc, %v2, %ct : (!cc, !ct_L5, !ct_L5) -> !ct_L5
  // CHECK: %[[V4:.*]] = openfhe.mul
  %v4 = openfhe.mul %cc, %v3, %ct : (!cc, !ct_L5, !ct_L5) -> !ct_L5
  // CHECK: %[[V5:.*]] = openfhe.mul
  %v5 = openfhe.mul %cc, %v4, %ct : (!cc, !ct_L5, !ct_L5) -> !ct_L5
  // At this point, cost = 5 * 10 = 50, which meets the threshold

  // This constant is defined after the prefetch
  // CHECK: %[[C5:.*]] = arith.constant 5
  %c5 = arith.constant 5 : i64

  // The load_key should use the original constant %c5
  // CHECK: %[[RK:.*]] = kmrt.load_key %[[C5]]
  %rk = kmrt.load_key %c5 : i64 -> !rk

  // Rotate the ciphertext
  // CHECK: %[[CT_ROT:.*]] = openfhe.rot %[[CC]], %[[V5]], %[[RK]]
  %ct_rot = openfhe.rot %cc, %v5, %rk : (!cc, !ct_L5, !rk) -> !ct_L5

  // Clear the key after use
  // CHECK: kmrt.clear_key %[[RK]]
  kmrt.clear_key %rk : !rk

  // CHECK: return %[[CT_ROT]]
  return %ct_rot : !ct_L5
}
