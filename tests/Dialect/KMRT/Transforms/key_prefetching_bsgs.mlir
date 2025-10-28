// RUN: heir-opt --kmrt-key-prefetching="prefetch-threshold=30" %s | FileCheck %s

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

// Test key prefetching with BSGS decomposition (nested loops)
// The pass should insert prefetches for both outer loop (giant steps) and inner loop (baby steps)

// CHECK-LABEL: func.func @bsgs_key_prefetching
// CHECK-SAME: (%[[CC:.*]]: !cc, %[[CT:.*]]: !ct_L5)
func.func @bsgs_key_prefetching(%cc: !cc, %ct: !ct_L5) -> !ct_L5 {
  %cst = arith.constant dense<[
    [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0],
    [5.0, 5.0, 5.0, 5.0], [6.0, 6.0, 6.0, 6.0], [7.0, 7.0, 7.0, 7.0], [8.0, 8.0, 8.0, 8.0],
    [9.0, 9.0, 9.0, 9.0], [10.0, 10.0, 10.0, 10.0], [11.0, 11.0, 11.0, 11.0], [12.0, 12.0, 12.0, 12.0],
    [13.0, 13.0, 13.0, 13.0], [14.0, 14.0, 14.0, 14.0], [15.0, 15.0, 15.0, 15.0], [16.0, 16.0, 16.0, 16.0]
  ]> : tensor<16x4xf64>

  %extracted_slice = tensor.extract_slice %cst[0, 0] [1, 4] [1, 1] : tensor<16x4xf64> to tensor<4xf64>
  %pt = lwe.rlwe_encode %extracted_slice {encoding = #inverse_canonical_encoding1, ring = #ring_f64_1} : tensor<4xf64> -> !pt
  %ct_init = openfhe.mul_plain %cc, %ct, %pt : (!cc, !ct_L5, !pt) -> !ct_L5

  %c3 = arith.constant {bsgs.tunable_param = "baby_step_size"} 3 : index

  // Outer loop (giant steps): iterates 0 to 4 (5 iterations)
  // Giant step indices: 0*3=0, 1*3=3, 2*3=6, 3*3=9, 4*3=12
  // With threshold=30 and loop cost ~26, prefetch distance = 1
  // Prefetch giant step key 0 before the outer loop

  // CHECK: %[[C0_OUTER:.*]] = arith.constant 0 : i64
  // CHECK-NEXT: kmrt.prefetch_key %[[C0_OUTER]]

  // CHECK: %[[RESULT:.*]] = affine.for %[[OUTER_IV:.*]] = 0 to
  %result = affine.for %outer_iv = 0 to #map()[%c3] iter_args(%acc_outer = %ct_init) -> (!ct_L5) {
    // Prefetch next giant step iteration (outer_iv + 1) inside the outer loop
    // This is placed at the beginning of the outer loop body
    // The prefetch computes the key index for the next iteration: (outer_iv + 1) * step
    // CHECK: affine.if #{{.*}}(%[[OUTER_IV]])[%c3] {
    // CHECK:   %[[NEXT_OUTER_IV:.*]] = affine.apply
    // CHECK:   %[[NEXT_GIANT_IDX:.*]] = affine.apply #{{.*}}(%[[NEXT_OUTER_IV]])[%c3]
    // CHECK:   %[[NEXT_GIANT_I64:.*]] = arith.index_cast %[[NEXT_GIANT_IDX]]
    // CHECK:   kmrt.prefetch_key %[[NEXT_GIANT_I64]]
    // CHECK: }

    // Load giant step key
    // CHECK: %[[GIANT_IDX:.*]] = affine.apply #{{.*}}(%[[OUTER_IV]])
    %giant_idx = affine.apply #map1(%outer_iv)[%c3]
    // CHECK: %[[GIANT_KEY:.*]] = kmrt.load_key %[[GIANT_IDX]]
    %giant_key = kmrt.load_key %giant_idx : index -> <>
    // CHECK: %[[CT_GIANT:.*]] = openfhe.rot %[[CC]], %[[CT]], %[[GIANT_KEY]]
    %ct_giant = openfhe.rot %cc, %ct, %giant_key : (!cc, !ct_L5, !kmrt.rot_key<>) -> !ct_L5
    // CHECK: kmrt.clear_key %[[GIANT_KEY]]
    kmrt.clear_key %giant_key : <>

    // Inner loop (baby steps): iterates 0 to 2 (3 iterations)
    // Prefetch baby step key 0 before the inner loop

    // CHECK: %[[C0_INNER:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: kmrt.prefetch_key %[[C0_INNER]]

    // CHECK: %[[RESULT_INNER:.*]] = affine.for %[[INNER_IV:.*]] = 0 to
    %result_inner = affine.for %inner_iv = 0 to #map2(%c3) iter_args(%acc_inner = %acc_outer) -> (!ct_L5) {
      // Prefetch next baby step iteration (inner_iv + 1) inside the inner loop
      // For baby steps, the index is just inner_iv (identity map), so prefetch is just inner_iv + 1
      // CHECK: affine.if #{{.*}}(%[[INNER_IV]], %c3) {
      // CHECK:   %[[NEXT_INNER_IV:.*]] = affine.apply
      // CHECK:   %[[NEXT_INNER_I64:.*]] = arith.index_cast %[[NEXT_INNER_IV]]
      // CHECK:   kmrt.prefetch_key %[[NEXT_INNER_I64]]
      // CHECK: }

      // CHECK: %[[BABY_KEY:.*]] = kmrt.load_key %[[INNER_IV]]
      %baby_key = kmrt.load_key %inner_iv : index -> <>
      // CHECK: %[[CT_BABY:.*]] = openfhe.rot %[[CC]], %[[CT_GIANT]], %[[BABY_KEY]]
      %ct_baby = openfhe.rot %cc, %ct_giant, %baby_key : (!cc, !ct_L5, !kmrt.rot_key<>) -> !ct_L5
      // CHECK: kmrt.clear_key %[[BABY_KEY]]
      kmrt.clear_key %baby_key : <>

      %combined_idx = affine.apply #map3(%outer_iv, %inner_iv)[%c3]
      %extracted_slice_2 = tensor.extract_slice %cst[%combined_idx, 0] [1, 4] [1, 1] : tensor<16x4xf64> to tensor<4xf64>
      %pt_2 = lwe.rlwe_encode %extracted_slice_2 {encoding = #inverse_canonical_encoding1, ring = #ring_f64_1} : tensor<4xf64> -> !pt
      %ct_mul = openfhe.mul_plain %cc, %ct_baby, %pt_2 : (!cc, !ct_L5, !pt) -> !ct_L5
      %ct_add = openfhe.add %cc, %acc_inner, %ct_mul : (!cc, !ct_L5, !ct_L5) -> !ct_L5
      affine.yield %ct_add : !ct_L5
    }
    affine.yield %result_inner : !ct_L5
  }

  // CHECK: return %[[RESULT]]
  return %result : !ct_L5
}
