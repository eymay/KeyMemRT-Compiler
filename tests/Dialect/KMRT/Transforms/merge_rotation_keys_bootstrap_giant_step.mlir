// RUN: keymemrt-opt --bootstrap-rotation-analysis --kmrt-merge-rotation-keys %s | FileCheck %s

// This test exercises the merge between bootstrap-loaded rotation keys and the
// giant-step rotation keys emitted (by hand here, mimicking the output of
// --symbolic-bsgs-decomposition) inside a BSGS-decomposed linear transform's
// outer loop.
//
// The pre-rotation by 32 plus the bootstrap make rotation key 32 resident
// at the bootstrap's "kept" boundary. The outer affine.for that follows
// rotates by (g * 32) for g in [1, 4): so the first iteration (g==1) wants
// to rotate by 32 as well. Before the fix, the merge pass treats the bare
// giant-step load as invisible and emits a redundant load/clear of key 32
// at g==1. After the fix the merge pass guards the bare giant-step load
// with an affine.if that reuses the bootstrap-resident key on the first
// iteration and falls through to a fresh kmrt.load_key on later iterations.

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

#giant_step_map = affine_map<(d0) -> (d0 * 32)>

!cc = !openfhe.crypto_context
!ct = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0, key = #key, modulus_chain = #modulus_chain_L5_C0>
!rk = !kmrt.rot_key<>
!rk32 = !kmrt.rot_key<rotation_index = 32>

// CHECK-LABEL: func.func @test_bootstrap_then_giant_step_loop
func.func @test_bootstrap_then_giant_step_loop(%cc: !cc, %ct: !ct) -> !ct {
  openfhe.setup_bootstrap %cc {levelBudgetDecode = 3 : index, levelBudgetEncode = 3 : index} : (!cc) -> ()

  // Rotation with index 32 before bootstrap.
  // CHECK: arith.constant 32
  %c32 = arith.constant 32 : i64
  // CHECK: %[[RK32:.*]] = kmrt.load_key {{.*}} : i64 -> !rk{{[0-9]*}}{{$}}
  %rk_pre = kmrt.load_key %c32 : i64 -> !rk32
  // CHECK: openfhe.rot {{.*}}, {{.*}}, %[[RK32]]
  %ct_pre = openfhe.rot %cc, %ct, %rk_pre : (!cc, !ct, !rk32) -> !ct
  // Pre-bootstrap clear of key 32 should be removed by the merge pass: bootstrap
  // also loads key 32 (it is one of the resident bootstrap keys), and the merge
  // pass should keep the key live across the bootstrap.
  // CHECK-NOT: kmrt.clear_key %[[RK32]]
  kmrt.clear_key %rk_pre : !rk32

  // CHECK: openfhe.bootstrap
  %ct_boot = openfhe.bootstrap %cc, %ct_pre : (!cc, !ct) -> !ct

  // After bootstrap, the giant-step outer loop wants to rotate by g*32 for
  // g in [1, 4): {32, 64, 96}. On iteration g==1 the giant-step key matches
  // the bootstrap-resident key 32 and should be reused via affine.if; on
  // other iterations a fresh kmrt.load_key fires.
  //
  // CHECK: affine.for %[[G:[a-zA-Z_0-9]+]] = 1 to 4
  // CHECK:   affine.if
  // CHECK-NEXT:   kmrt.use_key %[[RK32]]
  // CHECK:   } else {
  // CHECK-NEXT:   kmrt.load_key
  // CHECK:   openfhe.rot
  // The clear of the giant-step key must also be guarded so it doesn't
  // clobber the bootstrap-resident key on the matching iteration.
  // CHECK:   affine.if
  // CHECK:   } else {
  // CHECK-NEXT:   kmrt.clear_key
  %res = affine.for %g = 1 to 4 iter_args(%acc = %ct_boot) -> (!ct) {
    %idx = affine.apply #giant_step_map(%g)
    %idx_i64 = arith.index_cast %idx : index to i64
    %rk_giant = kmrt.load_key %idx_i64 : i64 -> !rk
    %ct_rot = openfhe.rot %cc, %acc, %rk_giant : (!cc, !ct, !rk) -> !ct
    kmrt.clear_key %rk_giant : !rk
    affine.yield %ct_rot : !ct
  }

  // CHECK: return
  return %res : !ct
}
