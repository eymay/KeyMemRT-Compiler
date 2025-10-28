// RUN: heir-opt --openfhe-configure-crypto-context=entry-function=linear_transform %s | FileCheck %s

// This test uses the output of BSGS decomposition directly to verify that
// ConfigureCryptoContext can find all rotation indices from dynamic rotation keys.

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
#map = affine_map<(d0)[s0] -> (d0 * s0)>
#map1 = affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>
#map2 = affine_map<(d0) -> (d0 + 12)>
#modulus_chain_L5_C5 = #lwe.modulus_chain<elements = <536903681 : i64, 67043329 : i64, 66994177 : i64, 67239937 : i64, 66961409 : i64, 66813953 : i64>, current = 5>
#ring_f64_1 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1>>
#ring_f64_1_x8192 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**8192>>
!rns_L5 = !rns.rns<!Z536903681_i64, !Z67043329_i64, !Z66994177_i64, !Z67239937_i64, !Z66961409_i64, !Z66813953_i64>
!pt = !lwe.new_lwe_plaintext<application_data = <message_type = tensor<4xf64>>, plaintext_space = <ring = #ring_f64_1, encoding = #inverse_canonical_encoding1>>
#ring_rns_L5_1_x8192 = #polynomial.ring<coefficientType = !rns_L5, polynomialModulus = <1 + x**8192>>
#ciphertext_space_L5 = #lwe.ciphertext_space<ring = #ring_rns_L5_1_x8192, encryption_type = mix>
!ct_L5 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<4xf64>>, plaintext_space = <ring = #ring_f64_1_x8192, encoding = #inverse_canonical_encoding>, ciphertext_space = #ciphertext_space_L5, key = #key, modulus_chain = #modulus_chain_L5_C5>

module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [536903681, 67043329, 66994177, 67239937, 66961409, 66813953], P = [67108864], logDefaultScale = 26>} {
  func.func @linear_transform(%cc: !cc, %ct: !ct_L5) -> !ct_L5 {
    %cst = arith.constant dense<[[1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00], [2.000000e+00, 2.000000e+00, 2.000000e+00, 2.000000e+00], [3.000000e+00, 3.000000e+00, 3.000000e+00, 3.000000e+00], [4.000000e+00, 4.000000e+00, 4.000000e+00, 4.000000e+00], [5.000000e+00, 5.000000e+00, 5.000000e+00, 5.000000e+00], [6.000000e+00, 6.000000e+00, 6.000000e+00, 6.000000e+00], [7.000000e+00, 7.000000e+00, 7.000000e+00, 7.000000e+00], [8.000000e+00, 8.000000e+00, 8.000000e+00, 8.000000e+00], [9.000000e+00, 9.000000e+00, 9.000000e+00, 9.000000e+00], [1.000000e+01, 1.000000e+01, 1.000000e+01, 1.000000e+01], [1.100000e+01, 1.100000e+01, 1.100000e+01, 1.100000e+01], [1.200000e+01, 1.200000e+01, 1.200000e+01, 1.200000e+01], [1.300000e+01, 1.300000e+01, 1.300000e+01, 1.300000e+01], [1.400000e+01, 1.400000e+01, 1.400000e+01, 1.400000e+01], [1.500000e+01, 1.500000e+01, 1.500000e+01, 1.500000e+01], [1.600000e+01, 1.600000e+01, 1.600000e+01, 1.600000e+01]]> : tensor<16x4xf64>
    %extracted_slice = tensor.extract_slice %cst[0, 0] [1, 4] [1, 1] : tensor<16x4xf64> to tensor<4xf64>
    %pt = lwe.rlwe_encode %extracted_slice {encoding = #inverse_canonical_encoding1, ring = #ring_f64_1} : tensor<4xf64> -> !pt
    %ct_0 = openfhe.mul_plain %cc, %ct, %pt : (!cc, !ct_L5, !pt) -> !ct_L5
    %c4 = arith.constant {bsgs.tunable_param = "baby_step_size"} 4 : index
    %ct_1 = affine.for %arg0 = 0 to 3 iter_args(%ct_4 = %ct_0) -> (!ct_L5) {
      %1 = affine.apply #map(%arg0)[%c4]
      %2 = kmrt.load_key %1 : index -> <>
      %ct_5 = openfhe.rot %cc, %ct, %2 : (!cc, !ct_L5, !kmrt.rot_key<>) -> !ct_L5
      kmrt.clear_key %2 : <>
      %ct_6 = affine.for %arg1 = 0 to 4 iter_args(%ct_7 = %ct_4) -> (!ct_L5) {
        %3 = affine.apply #map1(%arg0, %arg1)[%c4]
        %4 = kmrt.load_key %arg1 : index -> <>
        %ct_8 = openfhe.rot %cc, %ct_5, %4 : (!cc, !ct_L5, !kmrt.rot_key<>) -> !ct_L5
        kmrt.clear_key %4 : <>
        %extracted_slice_9 = tensor.extract_slice %cst[%3, 0] [1, 4] [1, 1] : tensor<16x4xf64> to tensor<4xf64>
        %pt_10 = lwe.rlwe_encode %extracted_slice_9 {encoding = #inverse_canonical_encoding1, ring = #ring_f64_1} : tensor<4xf64> -> !pt
        %ct_11 = openfhe.mul_plain %cc, %ct_8, %pt_10 : (!cc, !ct_L5, !pt) -> !ct_L5
        %ct_12 = openfhe.add %cc, %ct_7, %ct_11 : (!cc, !ct_L5, !ct_L5) -> !ct_L5
        affine.yield %ct_12 : !ct_L5
      }
      affine.yield %ct_6 : !ct_L5
    }
    %c12 = arith.constant 12 : index
    %0 = kmrt.load_key %c12 : index -> <>
    %ct_2 = openfhe.rot %cc, %ct, %0 : (!cc, !ct_L5, !kmrt.rot_key<>) -> !ct_L5
    kmrt.clear_key %0 : <>
    %ct_3 = affine.for %arg0 = 0 to 3 iter_args(%ct_4 = %ct_1) -> (!ct_L5) {
      %1 = affine.apply #map2(%arg0)
      %2 = kmrt.load_key %arg0 : index -> <>
      %ct_5 = openfhe.rot %cc, %ct_2, %2 : (!cc, !ct_L5, !kmrt.rot_key<>) -> !ct_L5
      kmrt.clear_key %2 : <>
      %extracted_slice_6 = tensor.extract_slice %cst[%1, 0] [1, 4] [1, 1] : tensor<16x4xf64> to tensor<4xf64>
      %pt_7 = lwe.rlwe_encode %extracted_slice_6 {encoding = #inverse_canonical_encoding1, ring = #ring_f64_1} : tensor<4xf64> -> !pt
      %ct_8 = openfhe.mul_plain %cc, %ct_5, %pt_7 : (!cc, !ct_L5, !pt) -> !ct_L5
      %ct_9 = openfhe.add %cc, %ct_4, %ct_8 : (!cc, !ct_L5, !ct_L5) -> !ct_L5
      affine.yield %ct_9 : !ct_L5
    }
    return %ct_3 : !ct_L5
  }
}

// CHECK: @linear_transform__configure_crypto_context
// CHECK: openfhe.gen_rotkey
// CHECK-SAME: array<i64: 0, 1, 2, 3, 4, 8, 12>
//
// With baby_step_size=4 and rotation range 1-15 (from original loop), BSGS creates:
// - Full giants (outer loop 0-2): d0 * 4 where d0 ∈ {0, 1, 2} → {0, 4, 8}
// - Baby steps (inner loop): d1 where d1 ∈ {0, 1, 2, 3} → {0, 1, 2, 3}
// - Remainder giant: 12
// - Combined unique rotation indices needed: {0, 1, 2, 3, 4, 8, 12}
