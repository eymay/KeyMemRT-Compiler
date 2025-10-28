// RUN: heir-opt --symbolic-bsgs-decomposition --kmrt-merge-rotation-keys %s | FileCheck %s

!Z536903681_i64 = !mod_arith.int<536903681 : i64>
!Z66813953_i64 = !mod_arith.int<66813953 : i64>
!Z66961409_i64 = !mod_arith.int<66961409 : i64>
!Z66994177_i64 = !mod_arith.int<66994177 : i64>
!Z67043329_i64 = !mod_arith.int<67043329 : i64>
!Z67239937_i64 = !mod_arith.int<67239937 : i64>
!cc = !openfhe.crypto_context
!rk = !kmrt.rot_key<>
!rk2 = !kmrt.rot_key<rotation_index = 2>
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

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [536903681, 67043329, 66994177, 67239937, 66961409, 66813953], P = [67108864], logDefaultScale = 26>} {
  // CHECK-LABEL: @linear_transform_with_prerot
  func.func @linear_transform_with_prerot(%cc: !cc, %ct: !ct_L5) -> !ct_L5 {
      %cst = arith.constant dense<[
      [1.0, 1.0, 1.0, 1.0],
      [2.0, 2.0, 2.0, 2.0],
      [3.0, 3.0, 3.0, 3.0],
      [4.0, 4.0, 4.0, 4.0],
      [5.0, 5.0, 5.0, 5.0],
      [6.0, 6.0, 6.0, 6.0],
      [7.0, 7.0, 7.0, 7.0],
      [8.0, 8.0, 8.0, 8.0],
      [9.0, 9.0, 9.0, 9.0],
      [10.0, 10.0, 10.0, 10.0],
      [11.0, 11.0, 11.0, 11.0],
      [12.0, 12.0, 12.0, 12.0],
      [13.0, 13.0, 13.0, 13.0],
      [14.0, 14.0, 14.0, 14.0],
      [15.0, 15.0, 15.0, 15.0],
      [16.0, 16.0, 16.0, 16.0]
    ]> : tensor<16x4xf64>

    // Add a rotation by 2 before the linear transform
    // CHECK: %[[C2:.*]] = arith.constant 2
    %c2 = arith.constant 2 : i64
    // CHECK: %[[RK2:.*]] = kmrt.load_key %[[C2]]
    %rk_prerot = kmrt.load_key %c2 : i64 -> !rk2
    // CHECK: %[[CT_PREROT:.*]] = openfhe.rot %{{.*}}, %{{.*}}, %[[RK2]]
    %ct_prerot = openfhe.rot %cc, %ct, %rk_prerot : (!cc, !ct_L5, !rk2) -> !ct_L5
    // The clear is removed - key will be reused in the nested loop after BSGS decomposition
    // CHECK-NOT: kmrt.clear_key %[[RK2]]
    kmrt.clear_key %rk_prerot : !rk2

    %extracted_slice = tensor.extract_slice %cst[0, 0] [1, 4] [1, 1] : tensor<16x4xf64> to tensor<4xf64>
    %pt = lwe.rlwe_encode %extracted_slice {encoding = #inverse_canonical_encoding1, ring = #ring_f64_1} : tensor<4xf64> -> !pt
    %ct_0 = openfhe.mul_plain %cc, %ct, %pt : (!cc, !ct_L5, !pt) -> !ct_L5

    // The linear transform will be decomposed with BSGS
    // After BSGS, the inner loop should reuse the pre-loaded key 2
    // CHECK: affine.for
    // CHECK:   kmrt.load_key
    // CHECK:   openfhe.rot
    // CHECK:   kmrt.clear_key
    // CHECK:   affine.for
    // Inner loop load is wrapped with affine.if to reuse pre-loaded key
    // CHECK:     affine.if
    // CHECK-NEXT:   kmrt.use_key %[[RK2]]
    // CHECK:     } else {
    // CHECK-NEXT:   kmrt.load_key
    %ct_1 = affine.for %arg0 = 1 to 16 iter_args(%ct_2 = %ct_0) -> (!ct_L5) {
      %0 = arith.index_cast %arg0 : index to i64
      %rk_loop = kmrt.load_key %0 : i64 -> !rk
      %ct_3 = openfhe.rot %cc, %ct, %rk_loop : (!cc, !ct_L5, !rk) -> !ct_L5
      kmrt.clear_key %rk_loop : !rk
      %extracted_slice_4 = tensor.extract_slice %cst[%arg0, 0] [1, 4] [1, 1] : tensor<16x4xf64> to tensor<4xf64>
      %pt_5 = lwe.rlwe_encode %extracted_slice_4 {encoding = #inverse_canonical_encoding1, ring = #ring_f64_1} : tensor<4xf64> -> !pt
      %ct_6 = openfhe.mul_plain %cc, %ct_3, %pt_5 : (!cc, !ct_L5, !pt) -> !ct_L5
      %ct_7 = openfhe.add %cc, %ct_2, %ct_6 : (!cc, !ct_L5, !ct_L5) -> !ct_L5
      affine.yield %ct_7 : !ct_L5
    }
    return %ct_1 : !ct_L5
  }
}
