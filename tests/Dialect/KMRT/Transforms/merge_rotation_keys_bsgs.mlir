// RUN: heir-opt --symbolic-bsgs-decomposition --kmrt-merge-rotation-keys %s | FileCheck %s

!Z536903681_i64 = !mod_arith.int<536903681 : i64>
!Z66813953_i64 = !mod_arith.int<66813953 : i64>
!Z66961409_i64 = !mod_arith.int<66961409 : i64>
!Z66994177_i64 = !mod_arith.int<66994177 : i64>
!Z67043329_i64 = !mod_arith.int<67043329 : i64>
!Z67239937_i64 = !mod_arith.int<67239937 : i64>
!cc = !openfhe.crypto_context
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 26>
#inverse_canonical_encoding1 = #lwe.inverse_canonical_encoding<scaling_factor = 1>
#key = #lwe.key<>
#map = affine_map<()[s0] -> ((s0 + 14) floordiv s0)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>
#modulus_chain_L5_C5 = #lwe.modulus_chain<elements = <536903681 : i64, 67043329 : i64, 66994177 : i64, 67239937 : i64, 66961409 : i64, 66813953 : i64>, current = 5>
#ring_f64_1 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1>>
#ring_f64_1_x8192 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**8192>>
!rns_L5 = !rns.rns<!Z536903681_i64, !Z67043329_i64, !Z66994177_i64, !Z67239937_i64, !Z66961409_i64, !Z66813953_i64>
!pt = !lwe.new_lwe_plaintext<application_data = <message_type = tensor<4xf64>>, plaintext_space = <ring = #ring_f64_1, encoding = #inverse_canonical_encoding1>>
#ring_rns_L5_1_x8192 = #polynomial.ring<coefficientType = !rns_L5, polynomialModulus = <1 + x**8192>>
#ciphertext_space_L5 = #lwe.ciphertext_space<ring = #ring_rns_L5_1_x8192, encryption_type = mix>
!ct_L5 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<4xf64>>, plaintext_space = <ring = #ring_f64_1_x8192, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L5, key = #key, modulus_chain = #modulus_chain_L5_C5>

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [536903681, 67043329, 66994177, 67239937, 66961409, 66813953], P = [67108864], logDefaultScale = 26>} {
  // CHECK-LABEL: @nested_bsgs_key_reuse
  func.func @nested_bsgs_key_reuse(%cc: !cc, %ct: !ct_L5) -> !ct_L5 {
    %cst = arith.constant dense<[[1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00], [3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00], [4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00], [5.000000e+00, 5.000000e+00, 5.000000e+00, 5.000000e+00], [6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00], [7.000000e+00, 7.000000e+00, 7.000000e+00, 7.000000e+00], [8.000000e+00, 8.000000e+00, 8.000000e+00, 8.000000e+00], [9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00], [1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01], [1.100000e+01, 1.100000e+01, 1.100000e+01, 1.100000e+01], [1.200000e+01, 1.200000e+01, 1.200000e+01, 1.200000e+01], [1.300000e+01, 1.300000e+01, 1.300000e+01, 1.300000e+01], [1.400000e+01, 1.400000e+01, 1.400000e+01, 1.400000e+01], [1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01], [1.600000e+01, 1.600000e+01, 1.600000e+01, 1.600000e+01]]> : tensor<16x4xf64>

    %extracted_slice = tensor.extract_slice %cst[0, 0] [1, 4] [1, 1] : tensor<16x4xf64> to tensor<4xf64>
    %pt = lwe.rlwe_encode %extracted_slice {encoding = #inverse_canonical_encoding1, ring = #ring_f64_1} : tensor<4xf64> -> !pt
    %ct_0 = openfhe.mul_plain %cc, %ct, %pt : (!cc, !ct_L5, !pt) -> !ct_L5

    // BSGS decomposition creates nested loops with memref for key reuse
    // CHECK: memref.alloca() : memref<4x!rk>
    // Prologue loop to preload baby step keys (0 to 4)
    // CHECK: affine.for %{{.*}} = 0 to 4
    // CHECK:   kmrt.load_key
    // CHECK:   memref.store
    // Main BSGS outer loop (giant steps) from 1 to 2
    // CHECK: affine.for %{{.*}} = 1 to 2
    // CHECK:   kmrt.load_key
    // Main BSGS inner loop (baby steps) from 0 to 4
    // CHECK:   affine.for %{{.*}} = 0 to 4
    // CHECK:     memref.load
    // CHECK:     kmrt.use_key

    %ct_1 = affine.for %arg0 = 1 to 13 iter_args(%ct_2 = %ct_0) -> (!ct_L5) {
      %0 = arith.index_cast %arg0 : index to i64
      %rk = kmrt.load_key %0 : i64 -> <rotation_index = 0>
      %ct_3 = openfhe.rot %cc, %ct, %rk : (!cc, !ct_L5, !kmrt.rot_key<rotation_index = 0>) -> !ct_L5
      kmrt.clear_key %rk : <rotation_index = 0>
      %extracted_slice_4 = tensor.extract_slice %cst[%arg0, 0] [1, 4] [1, 1] : tensor<16x4xf64> to tensor<4xf64>
      %pt_5 = lwe.rlwe_encode %extracted_slice_4 {encoding = #inverse_canonical_encoding1, ring = #ring_f64_1} : tensor<4xf64> -> !pt
      %ct_6 = openfhe.mul_plain %cc, %ct_3, %pt_5 : (!cc, !ct_L5, !pt) -> !ct_L5
      %ct_7 = openfhe.add %cc, %ct_2, %ct_6 : (!cc, !ct_L5, !ct_L5) -> !ct_L5
      affine.yield %ct_7 : !ct_L5
    }

    // CHECK: return
    return %ct_1 : !ct_L5
  }
}
