// RUN: heir-translate %s --emit-openfhe-pke | FileCheck %s

#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !mod_arith.int<65537 : i64>, polynomialModulus = <1 + x**32>>
#rns_L0_ = #rns.rns<!mod_arith.int<1095233372161 : i64>>
#ring_rns_L0_1_x32_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**32>>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain_L5_C0_ = #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64, 1005037682689 : i64, 998595133441 : i64, 972824936449 : i64, 959939837953 : i64>, current = 0>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32_, encryption_type = lsb>
!cc = !openfhe.crypto_context
!ct = !lwe.new_lwe_ciphertext<application_data = <message_type = i3>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain_L5_C0_>

// CHECK: void test_static_rotation_keys(
// CHECK-SAME:    CryptoContextT [[CC:[^,]*]],
// CHECK-SAME:    CiphertextT [[CT:[^)]*]]
// CHECK-SAME:  ) {
// CHECK-NEXT:      [[maybe_unused]] int64_t [[idx5:.*]] = 5;
// CHECK-NEXT:      RotKey [[rk1:.*]] = keymem_rt.deserializeKey([[idx5]], 2);
// CHECK-NEXT:      auto [[ct1:.*]] = [[CC]]->EvalRotate([[CT]], [[idx5]]);
// CHECK-NEXT:      LOG_ROT([[CT]],"%ct");
// CHECK-NEXT:      LOG_CT([[ct1]],"%ct_0");
// CHECK-NEXT:      keymem_rt.clearKey([[rk1]]);
// CHECK-NEXT:      [[maybe_unused]] int64_t [[idx16:.*]] = 16;
// CHECK-NEXT:      RotKey [[rk2:.*]] = keymem_rt.deserializeKey([[idx16]], 4);
// CHECK-NEXT:      auto [[ct2:.*]] = [[CC]]->EvalRotate([[ct1]], [[idx16]]);
// CHECK-NEXT:      LOG_ROT([[ct1]],"%ct_0");
// CHECK-NEXT:      LOG_CT([[ct2]],"%ct_1");
// CHECK-NEXT:      keymem_rt.clearKey([[rk2]]);
// CHECK-NEXT:  }
module attributes {scheme.ckks} {
  func.func @test_static_rotation_keys(%cc: !cc, %ct: !ct) {
    %c5 = arith.constant 5 : i64
    %rk1 = kmrt.load_key %c5 {key_depth = 2 : i64} : i64 -> !kmrt.rot_key<rotation_index = 5>
    %ct_0 = openfhe.rot %cc, %ct, %rk1 : (!cc, !ct, !kmrt.rot_key<rotation_index = 5>) -> !ct
    kmrt.clear_key %rk1 : !kmrt.rot_key<rotation_index = 5>
    %c16 = arith.constant 16 : i64
    %rk2 = kmrt.load_key %c16 {key_depth = 4 : i64} : i64 -> !kmrt.rot_key<rotation_index = 16>
    %ct_1 = openfhe.rot %cc, %ct_0, %rk2 : (!cc, !ct, !kmrt.rot_key<rotation_index = 16>) -> !ct
    kmrt.clear_key %rk2 : !kmrt.rot_key<rotation_index = 16>
    return
  }
}

// -----

// CHECK: void test_dynamic_rotation_keys(
// CHECK-SAME:    CryptoContextT [[CC:[^,]*]],
// CHECK-SAME:    CiphertextT [[CT:[^,]*]],
// CHECK-SAME:    size_t [[IDX:[^)]*]]
// CHECK-SAME:  ) {
// CHECK-NEXT:      RotKey [[rk:.*]] = keymem_rt.deserializeKey([[IDX]]);
// CHECK-NEXT:      auto [[ct1:.*]] = [[CC]]->EvalRotate([[CT]], [[IDX]]);
// CHECK-NEXT:      LOG_ROT([[CT]],"%ct");
// CHECK-NEXT:      LOG_CT([[ct1]],"%ct_0");
// CHECK-NEXT:      keymem_rt.clearKey([[rk]]);
// CHECK-NEXT:  }
module attributes {scheme.ckks} {
  func.func @test_dynamic_rotation_keys(%cc: !cc, %ct: !ct, %idx: index) {
    %rk = kmrt.load_key %idx : index -> !kmrt.rot_key<>
    %ct_0 = openfhe.rot %cc, %ct, %rk : (!cc, !ct, !kmrt.rot_key<>) -> !ct
    kmrt.clear_key %rk : !kmrt.rot_key<>
    return
  }
}

// -----

// CHECK: void test_mixed_rotation_keys(
// CHECK-SAME:    CryptoContextT [[CC:[^,]*]],
// CHECK-SAME:    CiphertextT [[CT:[^,]*]],
// CHECK-SAME:    size_t [[IDX:[^)]*]]
// CHECK-SAME:  ) {
// CHECK-NEXT:      [[maybe_unused]] int64_t [[idx5:.*]] = 5;
// CHECK-NEXT:      RotKey [[rk_static:.*]] = keymem_rt.deserializeKey([[idx5]], 2);
// CHECK-NEXT:      auto [[ct1:.*]] = [[CC]]->EvalRotate([[CT]], [[idx5]]);
// CHECK-NEXT:      LOG_ROT([[CT]],"%ct");
// CHECK-NEXT:      LOG_CT([[ct1]],"%ct_0");
// CHECK-NEXT:      keymem_rt.clearKey([[rk_static]]);
// CHECK-NEXT:      RotKey [[rk_dynamic:.*]] = keymem_rt.deserializeKey([[IDX]]);
// CHECK-NEXT:      auto [[ct2:.*]] = [[CC]]->EvalRotate([[ct1]], [[IDX]]);
// CHECK-NEXT:      LOG_ROT([[ct1]],"%ct_0");
// CHECK-NEXT:      LOG_CT([[ct2]],"%ct_1");
// CHECK-NEXT:      keymem_rt.clearKey([[rk_dynamic]]);
// CHECK-NEXT:  }
module attributes {scheme.ckks} {
  func.func @test_mixed_rotation_keys(%cc: !cc, %ct: !ct, %idx: index) {
    %c5 = arith.constant 5 : i64
    %rk_static = kmrt.load_key %c5 {key_depth = 2 : i64} : i64 -> !kmrt.rot_key<rotation_index = 5>
    %ct_0 = openfhe.rot %cc, %ct, %rk_static : (!cc, !ct, !kmrt.rot_key<rotation_index = 5>) -> !ct
    kmrt.clear_key %rk_static : !kmrt.rot_key<rotation_index = 5>
    %rk_dynamic = kmrt.load_key %idx : index -> !kmrt.rot_key<>
    %ct_1 = openfhe.rot %cc, %ct_0, %rk_dynamic : (!cc, !ct, !kmrt.rot_key<>) -> !ct
    kmrt.clear_key %rk_dynamic : !kmrt.rot_key<>
    return
  }
}
