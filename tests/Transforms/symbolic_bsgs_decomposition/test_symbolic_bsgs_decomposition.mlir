// RUN: heir-opt --symbolic-bsgs-decomposition %s | FileCheck %s

!Z536903681_i64 = !mod_arith.int<536903681 : i64>
!Z66813953_i64 = !mod_arith.int<66813953 : i64>
!Z66961409_i64 = !mod_arith.int<66961409 : i64>
!Z66994177_i64 = !mod_arith.int<66994177 : i64>
!Z67043329_i64 = !mod_arith.int<67043329 : i64>
!Z67239937_i64 = !mod_arith.int<67239937 : i64>
!cc = !openfhe.crypto_context
!rk = !kmrt.rot_key<rotation_index = 0>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 26>
#inverse_canonical_encoding1 = #lwe.inverse_canonical_encoding<scaling_factor = 1>
#key = #lwe.key<>
// CHECK-DAG: #[[GIANT_STEP:.*]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: #[[ACTUAL_IDX:.*]] = affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>
// CHECK-DAG: #[[REM_IDX:.*]] = affine_map<(d0) -> (d0 + 12)>
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
    %extracted_slice = tensor.extract_slice %cst[0, 0] [1, 4] [1, 1] : tensor<16x4xf64> to tensor<4xf64>
    %pt = lwe.rlwe_encode %extracted_slice {encoding = #inverse_canonical_encoding1, ring = #ring_f64_1} : tensor<4xf64> -> !pt
    %ct_0 = openfhe.mul_plain %cc, %ct, %pt : (!cc, !ct_L5, !pt) -> !ct_L5
    // CHECK: %[[N2:.*]] = arith.constant {bsgs.tunable_param = "baby_step_size"} 4 : index
    // CHECK: affine.for %[[GIANT_IV:.*]] = 0 to 3
    // CHECK:   %[[GIANT_AMT:.*]] = affine.apply #[[GIANT_STEP]](%[[GIANT_IV]])[%[[N2]]]
    // CHECK:   %[[EK_GIANT:.*]] = kmrt.load_key %[[GIANT_AMT]] : index -> <>
    // CHECK:   %[[CT_GIANT:.*]] = openfhe.rot %cc, %ct, %[[EK_GIANT]]
    // CHECK:   kmrt.clear_key %[[EK_GIANT]] : <>
    // CHECK:   affine.for %[[BABY_IV:.*]] = 0 to 4
    // CHECK:     %[[ACTUAL_IV:.*]] = affine.apply #[[ACTUAL_IDX]](%[[GIANT_IV]], %[[BABY_IV]])[%[[N2]]]
    // CHECK:     %[[EK_BABY:.*]] = kmrt.load_key %[[BABY_IV]] : index -> <>
    // CHECK:     %[[CT_BABY:.*]] = openfhe.rot %cc, %[[CT_GIANT]], %[[EK_BABY]]
    // CHECK:     kmrt.clear_key %[[EK_BABY]] : <>
    // CHECK:     tensor.extract_slice
    // CHECK:     lwe.rlwe_encode
    // CHECK:     openfhe.mul_plain
    // CHECK:     openfhe.add
    // Remainder loop
    // CHECK: %[[C12:.*]] = arith.constant 12 : index
    // CHECK: %[[EK_REM_GIANT:.*]] = kmrt.load_key %[[C12]] : index -> <>
    // CHECK: %[[CT_REM_GIANT:.*]] = openfhe.rot %cc, %ct, %[[EK_REM_GIANT]]
    // CHECK: kmrt.clear_key %[[EK_REM_GIANT]] : <>
    // CHECK: affine.for %[[REM_IV:.*]] = 0 to 3
    // CHECK:   %[[REM_ACTUAL_IV:.*]] = affine.apply #[[REM_IDX]](%[[REM_IV]])
    // CHECK:   %[[EK_REM_BABY:.*]] = kmrt.load_key %[[REM_IV]] : index -> <>
    // CHECK:   %[[CT_REM_BABY:.*]] = openfhe.rot %cc, %[[CT_REM_GIANT]], %[[EK_REM_BABY]]
    // CHECK:   kmrt.clear_key %[[EK_REM_BABY]] : <>
    %ct_1 = affine.for %arg0 = 1 to 16 iter_args(%ct_2 = %ct_0) -> (!ct_L5) {
      %0 = arith.index_cast %arg0 : index to i64
      %rk = kmrt.load_key %0 : i64 -> !rk
      %ct_3 = openfhe.rot %cc, %ct, %rk : (!cc, !ct_L5, !rk) -> !ct_L5
      kmrt.clear_key %rk : !rk
      %extracted_slice_4 = tensor.extract_slice %cst[%arg0, 0] [1, 4] [1, 1] : tensor<16x4xf64> to tensor<4xf64>
      %pt_5 = lwe.rlwe_encode %extracted_slice_4 {encoding = #inverse_canonical_encoding1, ring = #ring_f64_1} : tensor<4xf64> -> !pt
      %ct_6 = openfhe.mul_plain %cc, %ct_3, %pt_5 : (!cc, !ct_L5, !pt) -> !ct_L5
      %ct_7 = openfhe.add %cc, %ct_2, %ct_6 : (!cc, !ct_L5, !ct_L5) -> !ct_L5
      affine.yield %ct_7 : !ct_L5
    }
    return %ct_1 : !ct_L5
  }
}
