// RUN: heir-opt --openfhe-fast-rotation-precompute %s | FileCheck %s

!Z1032955396097_i64 = !mod_arith.int<1032955396097 : i64>
!Z1095233372161_i64 = !mod_arith.int<1095233372161 : i64>
!Z65537_i64 = !mod_arith.int<65537 : i64>
!cc = !openfhe.crypto_context
!rk = !kmrt.rot_key<>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C1 = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 1>
!rns_L1 = !rns.rns<!Z1095233372161_i64, !Z1032955396097_i64>
#ring_Z65537_i64_1_x32 = #polynomial.ring<coefficientType = !Z65537_i64, polynomialModulus = <1 + x**32>>
#ring_rns_L1_1_x32 = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**32>>
#ciphertext_space_L1 = #lwe.ciphertext_space<ring = #ring_rns_L1_1_x32, encryption_type = lsb>
!ct_L1 = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<32xi16>>, plaintext_space = <ring = #ring_Z65537_i64_1_x32, encoding = #full_crt_packing_encoding>, ciphertext_space = #ciphertext_space_L1, key = #key, modulus_chain = #modulus_chain_L5_C1>

// Test simple affine loop with same ciphertext rotated multiple times
// The precomputation should be hoisted before the loop
// CHECK-LABEL: func.func @affine_loop_same_input
func.func @affine_loop_same_input(%cc: !cc, %ct: !ct_L1) -> !ct_L1 {
  // CHECK: %[[PRECOMPUTE:.*]] = openfhe.fast_rotation_precompute %{{.*}}, %{{.*}}
  // CHECK: affine.for
  %result = affine.for %iv = 0 to 4 iter_args(%acc = %ct) -> (!ct_L1) {
    %idx1 = arith.constant 1 : i64
    %rk1 = kmrt.load_key %idx1 : i64 -> !kmrt.rot_key<rotation_index = 1>
    // CHECK: openfhe.fast_rotation %{{.*}}, %{{.*}}, %{{.*}}, %[[PRECOMPUTE]]
    %ct_rot1 = openfhe.rot %cc, %ct, %rk1 : (!cc, !ct_L1, !kmrt.rot_key<rotation_index = 1>) -> !ct_L1

    %idx2 = arith.constant 2 : i64
    %rk2 = kmrt.load_key %idx2 : i64 -> !kmrt.rot_key<rotation_index = 2>
    // CHECK: openfhe.fast_rotation %{{.*}}, %{{.*}}, %{{.*}}, %[[PRECOMPUTE]]
    %ct_rot2 = openfhe.rot %cc, %ct, %rk2 : (!cc, !ct_L1, !kmrt.rot_key<rotation_index = 2>) -> !ct_L1

    %ct_add1 = openfhe.add %cc, %ct_rot1, %ct_rot2 : (!cc, !ct_L1, !ct_L1) -> !ct_L1
    %ct_add2 = openfhe.add %cc, %acc, %ct_add1 : (!cc, !ct_L1, !ct_L1) -> !ct_L1
    affine.yield %ct_add2 : !ct_L1
  }
  return %result : !ct_L1
}

// Test nested affine loops where the same ciphertext is rotated with fixed indices in inner loop
// The precomputation should be hoisted before the outer loop
// CHECK-LABEL: func.func @nested_affine_loop_same_input
func.func @nested_affine_loop_same_input(%cc: !cc, %ct: !ct_L1) -> !ct_L1 {
  // CHECK: %[[PRECOMPUTE:.*]] = openfhe.fast_rotation_precompute
  // CHECK-NEXT: %{{.*}} = affine.for %[[OUTER_IV:.*]] =
  %result = affine.for %outer_iv = 0 to 4 iter_args(%outer_acc = %ct) -> (!ct_L1) {
    // Inner loop rotates the same ciphertext with different fixed indices
    // CHECK: %{{.*}} = affine.for %[[INNER_IV:.*]] =
    %inner_result = affine.for %inner_iv = 0 to 3 iter_args(%inner_acc = %ct) -> (!ct_L1) {
      %idx1 = arith.constant 1 : i64
      %rk1 = kmrt.load_key %idx1 : i64 -> !kmrt.rot_key<rotation_index = 1>
      // CHECK: openfhe.fast_rotation %{{.*}}, %{{.*}}, %{{.*}}, %[[PRECOMPUTE]]
      %ct_rot1 = openfhe.rot %cc, %ct, %rk1 : (!cc, !ct_L1, !kmrt.rot_key<rotation_index = 1>) -> !ct_L1

      %idx3 = arith.constant 3 : i64
      %rk3 = kmrt.load_key %idx3 : i64 -> !kmrt.rot_key<rotation_index = 3>
      // CHECK: openfhe.fast_rotation %{{.*}}, %{{.*}}, %{{.*}}, %[[PRECOMPUTE]]
      %ct_rot3 = openfhe.rot %cc, %ct, %rk3 : (!cc, !ct_L1, !kmrt.rot_key<rotation_index = 3>) -> !ct_L1

      %ct_add1 = openfhe.add %cc, %ct_rot1, %ct_rot3 : (!cc, !ct_L1, !ct_L1) -> !ct_L1
      %ct_add2 = openfhe.add %cc, %inner_acc, %ct_add1 : (!cc, !ct_L1, !ct_L1) -> !ct_L1
      affine.yield %ct_add2 : !ct_L1
    }
    %ct_final_add = openfhe.add %cc, %outer_acc, %inner_result : (!cc, !ct_L1, !ct_L1) -> !ct_L1
    affine.yield %ct_final_add : !ct_L1
  }
  return %result : !ct_L1
}

// Test affine loop with different ciphertexts
// Loop-invariant ct2 gets optimized (1 rotation per iteration, 4 iterations)
// Loop-variant acc does not get optimized
// CHECK-LABEL: func.func @affine_loop_different_inputs
func.func @affine_loop_different_inputs(%cc: !cc, %ct1: !ct_L1, %ct2: !ct_L1) -> !ct_L1 {
  // CHECK: %[[PRECOMPUTE:.*]] = openfhe.fast_rotation_precompute %{{.*}}, %{{.*}}
  // CHECK: affine.for
  %result = affine.for %iv = 0 to 4 iter_args(%acc = %ct1) -> (!ct_L1) {
    %idx1 = arith.constant 1 : i64
    %rk1 = kmrt.load_key %idx1 : i64 -> !kmrt.rot_key<rotation_index = 1>
    // CHECK: openfhe.rot %{{.*}}, %{{.*}}, %{{.*}} :
    %ct_rot1 = openfhe.rot %cc, %acc, %rk1 : (!cc, !ct_L1, !kmrt.rot_key<rotation_index = 1>) -> !ct_L1

    %idx2 = arith.constant 2 : i64
    %rk2 = kmrt.load_key %idx2 : i64 -> !kmrt.rot_key<rotation_index = 2>
    // CHECK: openfhe.fast_rotation %{{.*}}, %{{.*}}, %{{.*}}, %[[PRECOMPUTE]]
    %ct_rot2 = openfhe.rot %cc, %ct2, %rk2 : (!cc, !ct_L1, !kmrt.rot_key<rotation_index = 2>) -> !ct_L1

    %ct_add = openfhe.add %cc, %ct_rot1, %ct_rot2 : (!cc, !ct_L1, !ct_L1) -> !ct_L1
    affine.yield %ct_add : !ct_L1
  }
  return %result : !ct_L1
}

// Test BSGS pattern: outer loop creates rotated ciphertext, inner loop rotates it
// This tests the case where inner loop rotations work on a value created in outer loop
// CHECK-LABEL: func.func @bsgs_pattern
func.func @bsgs_pattern(%cc: !cc, %ct: !ct_L1, %ct_init: !ct_L1) -> !ct_L1 {
  // CHECK: affine.for %[[OUTER_IV:.*]] =
  %result = affine.for %outer_iv = 0 to 3 iter_args(%outer_acc = %ct_init) -> (!ct_L1) {
    // Outer loop creates a rotated ciphertext (giant step in BSGS)
    %idx_outer = arith.constant 8 : i64
    %rk_outer = kmrt.load_key %idx_outer : i64 -> !kmrt.rot_key<rotation_index = 8>
    // CHECK: %[[CT_OUTER:.*]] = openfhe.{{.*}}rotation
    %ct_outer = openfhe.rot %cc, %ct, %rk_outer : (!cc, !ct_L1, !kmrt.rot_key<rotation_index = 8>) -> !ct_L1

    // Inner loop rotates ct_outer (baby steps in BSGS)
    // Since ct_outer is loop-invariant with respect to inner loop, precompute should be hoisted
    // CHECK: %[[PRECOMPUTE:.*]] = openfhe.fast_rotation_precompute %{{.*}}, %[[CT_OUTER]]
    // CHECK-NEXT: %{{.*}} = affine.for %[[INNER_IV:.*]] =
    %inner_result = affine.for %inner_iv = 0 to 4 iter_args(%inner_acc = %outer_acc) -> (!ct_L1) {
      %idx_inner = arith.constant 1 : i64
      %rk_inner = kmrt.load_key %idx_inner : i64 -> !kmrt.rot_key<rotation_index = 1>
      // CHECK: openfhe.fast_rotation %{{.*}}, %[[CT_OUTER]], %{{.*}}, %[[PRECOMPUTE]]
      %ct_inner = openfhe.rot %cc, %ct_outer, %rk_inner : (!cc, !ct_L1, !kmrt.rot_key<rotation_index = 1>) -> !ct_L1

      %ct_add = openfhe.add %cc, %inner_acc, %ct_inner : (!cc, !ct_L1, !ct_L1) -> !ct_L1
      affine.yield %ct_add : !ct_L1
    }

    affine.yield %inner_result : !ct_L1
  }
  return %result : !ct_L1
}
