// RUN: heir-opt --kmrt-merge-rotation-keys %s | FileCheck %s

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

    // Pre-load key 1 before the nested loops
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    %c1 = arith.constant 1 : index
    // CHECK: %[[RK1_PRE:.*]] = kmrt.load_key %[[C1]]
    %0 = kmrt.load_key %c1 : index -> <>
    // CHECK: %[[CT_0:.*]] = openfhe.rot %{{.*}}, %{{.*}}, %[[RK1_PRE]]
    %ct_0 = openfhe.rot %cc, %ct, %0 : (!cc, !ct_L5, !kmrt.rot_key<>) -> !ct_L5
    // The clear is removed - key will be reused in the nested loop
    // CHECK-NOT: kmrt.clear_key %[[RK1_PRE]]
    kmrt.clear_key %0 : <>

    %extracted_slice = tensor.extract_slice %cst[0, 0] [1, 4] [1, 1] : tensor<16x4xf64> to tensor<4xf64>
    %pt = lwe.rlwe_encode %extracted_slice {encoding = #inverse_canonical_encoding1, ring = #ring_f64_1} : tensor<4xf64> -> !pt
    %ct_1 = openfhe.mul_plain %cc, %ct_0, %pt : (!cc, !ct_L5, !pt) -> !ct_L5
    %c3 = arith.constant {bsgs.tunable_param = "baby_step_size"} 3 : index

    // Outer loop (giant step)
    // CHECK: %[[CT_2:.*]] = affine.for %[[OUTER_IV:.*]] =
    %ct_2 = affine.for %arg0 = 0 to #map()[%c3] iter_args(%ct_3 = %ct_1) -> (!ct_L5) {
      %1 = affine.apply #map1(%arg0)[%c3]
      %2 = kmrt.load_key %1 : index -> <>
      %ct_4 = openfhe.rot %cc, %ct, %2 : (!cc, !ct_L5, !kmrt.rot_key<>) -> !ct_L5
      kmrt.clear_key %2 : <>

      // Inner loop (baby step) - loads keys 0, 1, 2
      // When %arg0 == 0 and %arg1 == 1, should reuse pre-loaded key 1
      // CHECK:   affine.for
      %ct_5 = affine.for %arg1 = 0 to #map2(%c3) iter_args(%ct_6 = %ct_3) -> (!ct_L5) {
        %3 = affine.apply #map3(%arg0, %arg1)[%c3]

        // The load is wrapped with affine.if checking both loop IVs
        // CHECK:     affine.apply
        // CHECK:     affine.if
        // CHECK-NEXT:   kmrt.use_key %[[RK1_PRE]]
        // CHECK:     } else {
        // CHECK-NEXT:   kmrt.load_key
        // CHECK:     }
        %4 = kmrt.load_key %arg1 : index -> <>
        %ct_7 = openfhe.rot %cc, %ct_4, %4 : (!cc, !ct_L5, !kmrt.rot_key<>) -> !ct_L5
        // CHECK:     kmrt.clear_key
        kmrt.clear_key %4 : <>
        %extracted_slice_8 = tensor.extract_slice %cst[%3, 0] [1, 4] [1, 1] : tensor<16x4xf64> to tensor<4xf64>
        %pt_9 = lwe.rlwe_encode %extracted_slice_8 {encoding = #inverse_canonical_encoding1, ring = #ring_f64_1} : tensor<4xf64> -> !pt
        %ct_10 = openfhe.mul_plain %cc, %ct_7, %pt_9 : (!cc, !ct_L5, !pt) -> !ct_L5
        %ct_11 = openfhe.add %cc, %ct_6, %ct_10 : (!cc, !ct_L5, !ct_L5) -> !ct_L5
        affine.yield %ct_11 : !ct_L5
      }
      affine.yield %ct_5 : !ct_L5
    }

    // The test verifies that the pass successfully optimizes nested loops (BSGS pattern)
    // by wrapping inner loads with affine.if to reuse pre-loaded keys when loop IVs match

    // CHECK: return %[[CT_2]]
    return %ct_2 : !ct_L5
  }
}
