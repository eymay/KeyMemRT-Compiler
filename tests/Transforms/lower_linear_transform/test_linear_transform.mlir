// RUN: heir-opt --lower-linear-transform %s | FileCheck %s

!Z536903681_i64 = !mod_arith.int<536903681 : i64>
!Z66813953_i64 = !mod_arith.int<66813953 : i64>
!Z66961409_i64 = !mod_arith.int<66961409 : i64>
!Z66994177_i64 = !mod_arith.int<66994177 : i64>
!Z67043329_i64 = !mod_arith.int<67043329 : i64>
!Z67239937_i64 = !mod_arith.int<67239937 : i64>
!cc = !openfhe.crypto_context
!ek = !openfhe.eval_key<index = 2048 : i32>
!ek1 = !openfhe.eval_key<index = 1024 : i32>
!ek2 = !openfhe.eval_key<index = 512 : i32>
!ek3 = !openfhe.eval_key<index = 256 : i32>
!ek4 = !openfhe.eval_key<index = 128 : i32>
#inverse_canonical_encoding = #lwe.inverse_canonical_encoding<scaling_factor = 26>
#inverse_canonical_encoding1 = #lwe.inverse_canonical_encoding<scaling_factor = 104>
#inverse_canonical_encoding2 = #lwe.inverse_canonical_encoding<scaling_factor = 52>
#key = #lwe.key<>
#modulus_chain_L5_C5 = #lwe.modulus_chain<elements = <536903681 : i64, 67043329 : i64, 66994177 : i64, 67239937 : i64, 66961409 : i64, 66813953 : i64>, current = 5>
#ring_f64_1_x8192 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**8192>>
!rns_L5 = !rns.rns<!Z536903681_i64, !Z67043329_i64, !Z66994177_i64, !Z67239937_i64, !Z66961409_i64, !Z66813953_i64>
!pt = !lwe.new_lwe_plaintext<application_data = <message_type = tensor<4xf64>>, plaintext_space = <ring = #ring_f64_1_x8192, encoding = #inverse_canonical_encoding>>
#ring_rns_L5_1_x8192 = #polynomial.ring<coefficientType = !rns_L5, polynomialModulus = <1 + x**8192>>
#ciphertext_space_L5 = #lwe.ciphertext_space<ring = #ring_rns_L5_1_x8192, encryption_type = mix>
#ciphertext_space_L5_D3 = #lwe.ciphertext_space<ring = #ring_rns_L5_1_x8192, encryption_type = mix, size = 3>
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
    // CHECK: lwe.rlwe_encode
    // CHECK: openfhe.mul_plain
    // CHECK: affine.for
    // CHECK:   openfhe.deserialize_key_dynamic
    // CHECK:   openfhe.rot
    // CHECK:   openfhe.clear_key
    // CHECK:   openfhe.mul_plain
    // CHECK:   openfhe.add
    %ct_0 = openfhe.linear_transform %cc, %ct, %cst {diagonal_count = 16 : i32, slots = 4 : i32} : (!ct_L5, tensor<16x4xf64>) -> !ct_L5
    return %ct_0 : !ct_L5
    }
  }
