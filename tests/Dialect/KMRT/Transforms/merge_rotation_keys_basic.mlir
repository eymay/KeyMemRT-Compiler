// RUN: heir-opt --kmrt-merge-rotation-keys %s | FileCheck %s

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

// CHECK-LABEL: func.func @basic_merge
// CHECK-SAME: (%[[CC:.*]]: !cc, %[[CT:.*]]: !ct_L5, %[[CT_0:.*]]: !ct_L5)
func.func @basic_merge(%cc: !cc, %ct1: !ct_L5, %ct2: !ct_L5) -> (!ct_L5, !ct_L5) {
  // First rotation with index 5
  // CHECK: %[[C5:.*]] = arith.constant 5
  %c5 = arith.constant 5 : i64
  // CHECK: %[[RK:.*]] = kmrt.load_key %[[C5]]
  %rk1 = kmrt.load_key %c5 : i64 -> !rk5
  // CHECK: %[[CT_1:.*]] = openfhe.rot %[[CC]], %[[CT]], %[[RK]]
  %rot1 = openfhe.rot %cc, %ct1, %rk1 : (!cc, !ct_L5, !rk5) -> !ct_L5
  // CHECK-NOT: kmrt.clear_key
  kmrt.clear_key %rk1 : !rk5

  // Second rotation with the same index 5 (should reuse the key)
  // CHECK-NOT: kmrt.load_key
  %rk2 = kmrt.load_key %c5 : i64 -> !rk5
  // CHECK: %[[CT_2:.*]] = openfhe.rot %[[CC]], %[[CT_0]], %[[RK]]
  %rot2 = openfhe.rot %cc, %ct2, %rk2 : (!cc, !ct_L5, !rk5) -> !ct_L5
  // CHECK: kmrt.clear_key %[[RK]]
  kmrt.clear_key %rk2 : !rk5

  // CHECK: return %[[CT_1]], %[[CT_2]]
  return %rot1, %rot2 : !ct_L5, !ct_L5
}
