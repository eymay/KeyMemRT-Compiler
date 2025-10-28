// RUN: heir-opt --kmrt-merge-rotation-keys %s | FileCheck %s

!Z536903681_i64 = !mod_arith.int<536903681 : i64>
!Z66813953_i64 = !mod_arith.int<66813953 : i64>
!Z66961409_i64 = !mod_arith.int<66961409 : i64>
!Z66994177_i64 = !mod_arith.int<66994177 : i64>
!Z67043329_i64 = !mod_arith.int<67043329 : i64>
!Z67239937_i64 = !mod_arith.int<67239937 : i64>
!cc = !openfhe.crypto_context
!rk = !kmrt.rot_key<>
!rk4 = !kmrt.rot_key<rotation_index = 4>
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
  // CHECK-LABEL: @prepost_key_reuse
  func.func @prepost_key_reuse(%cc: !cc, %ct_init: !ct_L5) -> !ct_L5 {
    // Pre-load key 4 before the loop
    // CHECK: %[[C4:.*]] = arith.constant 4
    %c4 = arith.constant 4 : i64
    // CHECK: %[[RK4_PRELOAD:.*]] = kmrt.load_key %[[C4]]
    %rk4_preload = kmrt.load_key %c4 : i64 -> !rk4

    // Use it before the loop
    // CHECK: %[[CT_PRE:.*]] = openfhe.rot %{{.*}}, %{{.*}}, %[[RK4_PRELOAD]]
    %ct_pre = openfhe.rot %cc, %ct_init, %rk4_preload : (!cc, !ct_L5, !rk4) -> !ct_L5

    // The clear should be removed because key 4 will be reused in the loop
    // CHECK-NOT: kmrt.clear_key %[[RK4_PRELOAD]]
    kmrt.clear_key %rk4_preload : !rk4

    // Loop that loads keys dynamically (indices 1-10)
    // This loop will reuse key 4 (preloop optimization)
    // and will skip clearing key 5 (postloop optimization)
    // CHECK: affine.for %[[IV:.*]] = 1 to 11
    %ct_result = affine.for %iv = 1 to 11 iter_args(%ct = %ct_pre) -> (!ct_L5) {
      %iv_i64 = arith.index_cast %iv : index to i64

      // The load should be wrapped with affine.if to reuse pre-loaded key 4
      // CHECK: %[[LOOP_RK:.*]] = affine.if #{{.*}}(%[[IV]]) -> !kmrt.rot_key<> {
      // CHECK-NEXT: %[[USE_KEY_4:.*]] = kmrt.use_key %[[RK4_PRELOAD]]
      // CHECK-NEXT: affine.yield %[[USE_KEY_4]]
      // CHECK-NEXT: } else {
      // CHECK-NEXT: %[[LOADED_RK:.*]] = kmrt.load_key
      // CHECK-NEXT: affine.yield %[[LOADED_RK]]
      // CHECK-NEXT: }
      %rk = kmrt.load_key %iv_i64 : i64 -> !rk

      // CHECK: %[[CT_ROT:.*]] = openfhe.rot %{{.*}}, %{{.*}}, %[[LOOP_RK]]
      %ct_rotated = openfhe.rot %cc, %ct, %rk : (!cc, !ct_L5, !rk) -> !ct_L5

      // The clear should be wrapped with affine.if to skip when iv == 5
      // CHECK: affine.if #{{.*}}(%[[IV]]) {
      // CHECK-NEXT: } else {
      // CHECK-NEXT: kmrt.clear_key %[[LOOP_RK]]
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
