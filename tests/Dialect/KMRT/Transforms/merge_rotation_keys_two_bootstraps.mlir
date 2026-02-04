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

// Test that rotation keys shared between two bootstraps are merged.
// The bootstrap-rotation-analysis pass will insert load/clear for all bootstrap keys for each bootstrap.
// The merge-rotation-keys pass should merge the clear-load pairs in between the two bootstraps.

// CHECK-LABEL: func.func @test_two_bootstraps
func.func @test_two_bootstraps(%cc: !cc, %ct: !ct) -> !ct {
  openfhe.setup_bootstrap %cc {levelBudgetDecode = 3 : index, levelBudgetEncode = 3 : index} : (!cc) -> ()

  // First bootstrap will have 53 rotation keys loaded
  // CHECK-COUNT-53: kmrt.load_key

  // First bootstrap operation
  // CHECK: openfhe.bootstrap
  %ct_boot1 = openfhe.bootstrap %cc, %ct : (!cc, !ct) -> !ct

  // Between the two bootstraps, NO clear operations should happen
  // (all clears should be removed by the merge pass)
  // CHECK-NOT: kmrt.clear_key

  // Between the two bootstraps, NO load operations should happen
  // (all loads should be removed by the merge pass since keys are reused)
  // CHECK-NOT: kmrt.load_key

  // Second bootstrap - should reuse all keys from the first bootstrap
  // CHECK: openfhe.bootstrap
  %ct_boot2 = openfhe.bootstrap %cc, %ct_boot1 : (!cc, !ct) -> !ct

  // After the second bootstrap, all 53 keys should be cleared exactly once
  // CHECK-COUNT-53: kmrt.clear_key

  // CHECK: return
  return %ct_boot2 : !ct
}
