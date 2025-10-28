// RUN: heir-opt --kmrt-merge-rotation-keys %s | FileCheck %s

!Z536903681_i64 = !mod_arith.int<536903681 : i64>
!Z66813953_i64 = !mod_arith.int<66813953 : i64>
!Z66961409_i64 = !mod_arith.int<66961409 : i64>
!Z66994177_i64 = !mod_arith.int<66994177 : i64>
!Z67043329_i64 = !mod_arith.int<67043329 : i64>
!Z67239937_i64 = !mod_arith.int<67239937 : i64>
!cc = !openfhe.crypto_context
!rk = !kmrt.rot_key<>
!rk5 = !kmrt.rot_key<rotation_index = 5>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 26>
#key = #lwe.key<>
#modulus_chain_L5_C5 = #lwe.modulus_chain<elements = <536903681 : i64, 67043329 : i64, 66994177 : i64, 67239937 : i64, 66961409 : i64, 66813953 : i64>, current = 5>
#ring_f64_1_x8192 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**8192>>
!rns_L5 = !rns.rns<!Z536903681_i64, !Z67043329_i64, !Z66994177_i64, !Z67239937_i64, !Z66961409_i64, !Z66813953_i64>
#ring_rns_L5_1_x8192 = #polynomial.ring<coefficientType = !rns_L5, polynomialModulus = <1 + x**8192>>
#ciphertext_space_L5 = #lwe.ciphertext_space<ring = #ring_rns_L5_1_x8192, encryption_type = mix>
!ct_L5 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<4xf64>>, plaintext_space = <ring = #ring_f64_1_x8192, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L5, key = #key, modulus_chain = #modulus_chain_L5_C5>

module {
  // CHECK-LABEL: @postloop_key_reuse
  func.func @postloop_key_reuse(%cc: !cc, %ct_init: !ct_L5) -> !ct_L5 {
    // Simple loop that loads keys dynamically (indices 1-10)
    // CHECK: affine.for %[[IV:.*]] = 1 to 11
    %ct_result = affine.for %iv = 1 to 11 iter_args(%ct = %ct_init) -> (!ct_L5) {
      %iv_i64 = arith.index_cast %iv : index to i64

      // CHECK: %[[RK:.*]] = kmrt.load_key
      %rk = kmrt.load_key %iv_i64 : i64 -> !rk

      // CHECK: %[[CT_ROT:.*]] = openfhe.rot %{{.*}}, %{{.*}}, %[[RK]]
      %ct_rotated = openfhe.rot %cc, %ct, %rk : (!cc, !ct_L5, !rk) -> !ct_L5

      // The clear should be wrapped with affine.if to skip when iv == 5
      // CHECK: affine.if #{{.*}}(%[[IV]]) {
      // CHECK-NEXT: } else {
      // CHECK-NEXT: kmrt.clear_key %[[RK]]
      // CHECK-NEXT: }
      kmrt.clear_key %rk : !rk

      affine.yield %ct_rotated : !ct_L5
    }

    // After the loop, we want to use key 5
    // The pass should detect that key 5 was loaded in the loop
    // and replace the load with assume_loaded
    // CHECK: %[[C5:.*]] = arith.constant 5
    // CHECK-NOT: kmrt.load_key
    // CHECK: %[[RK5:.*]] = kmrt.assume_loaded %[[C5]]
    // CHECK: %[[CT_POST:.*]] = openfhe.rot %{{.*}}, %{{.*}}, %[[RK5]]
    // CHECK: kmrt.clear_key %[[RK5]]
    %c5 = arith.constant 5 : i64
    %rk5 = kmrt.load_key %c5 : i64 -> !rk5
    %ct_post = openfhe.rot %cc, %ct_result, %rk5 : (!cc, !ct_L5, !rk5) -> !ct_L5
    kmrt.clear_key %rk5 : !rk5

    return %ct_post : !ct_L5
  }
}
