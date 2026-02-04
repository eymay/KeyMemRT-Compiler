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
!rk16 = !kmrt.rot_key<rotation_index = 16>

// Test that rotation key 16 is loaded once before bootstrap,
// used before bootstrap, reused after bootstrap, then cleared once.
// The bootstrap-rotation-analysis pass will insert load/clear for all 53 bootstrap keys.
// The merge-rotation-keys pass should merge the user's key 16 with bootstrap's key 16.

// CHECK-LABEL: func.func @test_rotation_with_bootstrap
func.func @test_rotation_with_bootstrap(%cc: !cc, %ct: !ct) -> !ct {
  openfhe.setup_bootstrap %cc {levelBudgetDecode = 3 : index, levelBudgetEncode = 3 : index} : (!cc) -> ()

  // Rotation with index 16 before bootstrap
  // CHECK: arith.constant 16
  %c16 = arith.constant 16 : i64
  // CHECK: %[[RK16:.*]] = kmrt.load_key {{.*}} : i64 -> !rk{{$}}
  %rk1 = kmrt.load_key %c16 : i64 -> !rk16
  // CHECK: openfhe.rot {{.*}}, {{.*}}, %[[RK16]]
  %ct_rot = openfhe.rot %cc, %ct, %rk1 : (!cc, !ct, !rk16) -> !ct
  // First clear should be removed by merge pass (key will be reused after bootstrap)
  // CHECK-NOT: kmrt.clear_key %[[RK16]]
  kmrt.clear_key %rk1 : !rk16

  // Bootstrap will have many rotation keys loaded (including key 16)
  // Bootstrap key loads happen here
  // CHECK: kmrt.load_key
  // CHECK: openfhe.bootstrap
  %ct_boot = openfhe.bootstrap %cc, %ct_rot : (!cc, !ct) -> !ct

  // After bootstrap, all keys EXCEPT key 16 are cleared
  // Many kmrt.clear_key operations for other bootstrap keys
  // CHECK: kmrt.clear_key
  // Second load of key 16 should be removed - the rotation should use the same %[[RK16]]
  %rk2 = kmrt.load_key %c16 : i64 -> !rk16
  // CHECK: openfhe.rot {{.*}}, {{.*}}, %[[RK16]]
  %ct_rot2 = openfhe.rot %cc, %ct_boot, %rk2 : (!cc, !ct, !rk16) -> !ct
  // Final clear of key 16 happens at the end
  // CHECK: kmrt.clear_key %[[RK16]]
  kmrt.clear_key %rk2 : !rk16

  // CHECK: return
  return %ct_rot2 : !ct
}
