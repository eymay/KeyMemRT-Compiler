// RUN: heir-opt --bootstrap-rotation-analysis --kmrt-merge-rotation-keys %s | FileCheck %s

!Z1095233372161_i64 = !mod_arith.int<1095233372161 : i64>
!Z65537_i64 = !mod_arith.int<65537 : i64>

!rns_L0 = !rns.rns<!Z1095233372161_i64>

#ring_Z65537_i64_1_x32768 = #polynomial.ring<coefficientType = !Z65537_i64, polynomialModulus = <1 + x**32768>>
#ring_rns_L0_1_x32768 = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**32768>>

#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 1024>
#key = #lwe.key<>

#modulus_chain_L5_C0 = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>

#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32768, encoding = #inverse_canonical_encoding>

#ciphertext_space_L0 = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32768, encryption_type = lsb>

!pt = !lwe.new_lwe_plaintext<application_data = <message_type = f16>, plaintext_space = #plaintext_space>
!cc = !openfhe.crypto_context
!ct = !lwe.new_lwe_ciphertext<application_data = <message_type = f16>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L5_C0>

// Test pattern: bootstrap -> affine.for with rotations -> bootstrap
// The loop uses rotation indices 0, 16, 32, 48, ..., 496, 512 (i*16 for i=0 to 32)
// Bootstrap uses static keys: 16, 32, 48, 512, 1024, 1536, 2048, ...
//
// This test verifies advanced loop-aware merging with computed indices:
// 1. Bootstrap 1 loads 53 keys
// 2. Bootstrap 1 keeps FOUR keys that match loop indices: 16, 32, 48, 512
// 3. Loop uses nested affine.if to check if iteration matches any kept key
// 4. Loop loads fresh keys for iterations that don't match
// 5. Bootstrap 2 may reuse keys kept by the loop

// CHECK-LABEL: func.func @test_bootstrap_loop_bootstrap
func.func @test_bootstrap_loop_bootstrap(%cc: !cc, %ct: !ct) -> !ct {
  openfhe.setup_bootstrap %cc {levelBudgetDecode = 3 : index, levelBudgetEncode = 3 : index} : (!cc) -> ()

  // First bootstrap will have 53 rotation keys loaded
  // CHECK-COUNT-53: kmrt.load_key

  // First bootstrap operation
  // CHECK: openfhe.bootstrap
  %ct_boot1 = openfhe.bootstrap %cc, %ct : (!cc, !ct) -> !ct

  // After first bootstrap, FOUR keys are NOT cleared (indices: 16, 32, 48, 512)
  // 49 keys are cleared here (53 - 4 = 49)

  // Linear transformation using affine.for with rotations
  // Loop goes from 0 to 33, computing rotation indices i*16: 0, 16, 32, ..., 512
  // Using affine.apply to compute the index
  // CHECK: affine.for
  %result = affine.for %i = 0 to 33 iter_args(%ct_iter = %ct_boot1) -> (!ct) {
    // Compute rotation index using affine map: i*16
    %rot_index = affine.apply affine_map<(d0) -> (d0 * 16)>(%i)

    // Loop has conditional logic for FOUR bootstrap keys:
    // - For iteration 1 (rot index 16): use_key (reuse from bootstrap)
    // - For iteration 2 (rot index 32): use_key (reuse from bootstrap)
    // - For iteration 3 (rot index 48): use_key (reuse from bootstrap)
    // - For iteration 32 (rot index 512): use_key (reuse from bootstrap)
    // - For other iterations: load_key (load fresh)
    // CHECK: affine.if
    // CHECK-NEXT: kmrt.use_key
    // CHECK: else
    // CHECK: affine.if
    // CHECK-NEXT: kmrt.use_key
    // CHECK: else
    // CHECK: affine.if
    // CHECK-NEXT: kmrt.use_key
    // CHECK: else
    // CHECK: affine.if
    // CHECK-NEXT: kmrt.use_key
    // CHECK: else
    // CHECK-NEXT: kmrt.load_key
    %rk = kmrt.load_key %rot_index : index -> !kmrt.rot_key<>
    %ct_rotated = openfhe.rot %cc, %ct_iter, %rk : (!cc, !ct, !kmrt.rot_key<>) -> !ct

    // Loop has conditional clear: skip clearing for pre-loaded keys (16, 32, 48, 512)
    // CHECK: affine.if
    // CHECK: else
    // CHECK: affine.if
    // CHECK: else
    // CHECK: affine.if
    // CHECK: else
    // CHECK: affine.if
    // CHECK: else
    // CHECK-NEXT: kmrt.clear_key
    kmrt.clear_key %rk : !kmrt.rot_key<>

    affine.yield %ct_rotated : !ct
  }

  // Second bootstrap may reuse keys from the loop (if kept alive)
  // Other keys are loaded fresh

  // Second bootstrap operation
  // CHECK: openfhe.bootstrap
  %ct_boot2 = openfhe.bootstrap %cc, %result : (!cc, !ct) -> !ct

  // After second bootstrap, all 53 keys are cleared
  // CHECK-COUNT-53: kmrt.clear_key

  // CHECK: return
  return %ct_boot2 : !ct
}
