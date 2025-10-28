// RUN: heir-opt --openfhe-insert-clear-ops %s | FileCheck %s

!Z1095233372161_i64_ = !mod_arith.int<1095233372161 : i64>
!Z65537_i64_ = !mod_arith.int<65537 : i64>
#full_crt_packing_encoding = #lwe.full_crt_packing_encoding<scaling_factor = 0>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <1095233372161 : i64>, current = 0>
!rns_L0_ = !rns.rns<!Z1095233372161_i64_>
#ring_Z65537_i64_1_x32_ = #polynomial.ring<coefficientType = !Z65537_i64_, polynomialModulus = <1 + x**32>>
#plaintext_space = #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x32_, encoding = #full_crt_packing_encoding>
#ring_rns_L0_1_x32_ = #polynomial.ring<coefficientType = !rns_L0_, polynomialModulus = <1 + x**32>>
!pt = !lwe.new_lwe_plaintext<application_data = <message_type = tensor<32xi16>>, plaintext_space = #plaintext_space>
#ciphertext_space_L0_ = #lwe.ciphertext_space<ring = #ring_rns_L0_1_x32_, encryption_type = lsb>
!ct_L0_ = !lwe.new_lwe_ciphertext<application_data = <message_type = tensor<32xi16>>, plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space_L0_, key = #key, modulus_chain = #modulus_chain>

// Test that clear ops are inserted for values not defined in loops
func.func @test_clear_ops_simple(%arg0: !openfhe.crypto_context, %arg1: !ct_L0_, %arg2: !ct_L0_) -> !ct_L0_ {
  %0 = openfhe.add %arg0, %arg1, %arg2 : (!openfhe.crypto_context, !ct_L0_, !ct_L0_) -> !ct_L0_
  %1 = openfhe.mul %arg0, %0, %arg2 : (!openfhe.crypto_context, !ct_L0_, !ct_L0_) -> !ct_L0_
  return %1 : !ct_L0_
}

// CHECK-LABEL: @test_clear_ops_simple
// CHECK: %[[V0:.*]] = openfhe.add
// CHECK-NEXT: %[[V1:.*]] = openfhe.mul
// CHECK-NEXT: openfhe.clear_ct %[[V0]]
// CHECK-NEXT: return %[[V1]]

// Test that clear ops are NOT inserted for values defined inside affine loops
func.func @test_no_clear_in_affine_loop(%arg0: !openfhe.crypto_context, %arg1: !ct_L0_) -> !ct_L0_ {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index

  %result = affine.for %i = 0 to 4 iter_args(%iter = %arg1) -> !ct_L0_ {
    %0 = openfhe.add %arg0, %iter, %arg1 : (!openfhe.crypto_context, !ct_L0_, !ct_L0_) -> !ct_L0_
    affine.yield %0 : !ct_L0_
  }

  return %result : !ct_L0_
}

// CHECK-LABEL: @test_no_clear_in_affine_loop
// CHECK: affine.for
// CHECK: %[[V0:.*]] = openfhe.add
// CHECK-NOT: openfhe.clear_ct %[[V0]]
// CHECK: affine.yield %[[V0]]
// CHECK: return

// Test that clear ops are NOT inserted for values defined inside scf loops
func.func @test_no_clear_in_scf_loop(%arg0: !openfhe.crypto_context, %arg1: !ct_L0_) -> !ct_L0_ {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index

  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%iter = %arg1) -> !ct_L0_ {
    %0 = openfhe.add %arg0, %iter, %arg1 : (!openfhe.crypto_context, !ct_L0_, !ct_L0_) -> !ct_L0_
    scf.yield %0 : !ct_L0_
  }

  return %result : !ct_L0_
}

// CHECK-LABEL: @test_no_clear_in_scf_loop
// CHECK: scf.for
// CHECK: %[[V0:.*]] = openfhe.add
// CHECK-NOT: openfhe.clear_ct %[[V0]]
// CHECK: scf.yield %[[V0]]
// CHECK: return

// Test mixed case: clear ops outside loops but not inside
func.func @test_mixed_clear_ops(%arg0: !openfhe.crypto_context, %arg1: !ct_L0_, %arg2: !ct_L0_) -> !ct_L0_ {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index

  // This should get a clear op
  %pre_loop = openfhe.add %arg0, %arg1, %arg2 : (!openfhe.crypto_context, !ct_L0_, !ct_L0_) -> !ct_L0_

  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%iter = %pre_loop) -> !ct_L0_ {
    // This should NOT get a clear op (defined in loop)
    %0 = openfhe.add %arg0, %iter, %arg1 : (!openfhe.crypto_context, !ct_L0_, !ct_L0_) -> !ct_L0_
    scf.yield %0 : !ct_L0_
  }

  // This should get a clear op
  %post_loop = openfhe.mul %arg0, %result, %arg2 : (!openfhe.crypto_context, !ct_L0_, !ct_L0_) -> !ct_L0_

  return %post_loop : !ct_L0_
}

// CHECK-LABEL: @test_mixed_clear_ops
// CHECK: %[[PRE:.*]] = openfhe.add
// CHECK: %[[RESULT:.*]] = scf.for
// CHECK: %[[LOOP:.*]] = openfhe.add
// CHECK-NOT: openfhe.clear_ct %[[LOOP]]
// CHECK: scf.yield %[[LOOP]]
// CHECK-NEXT: }
// CHECK-NEXT: openfhe.clear_ct %[[PRE]]
// CHECK-NEXT: %[[POST:.*]] = openfhe.mul
// CHECK-NEXT: openfhe.clear_ct %[[RESULT]]
// CHECK-NEXT: return %[[POST]]
