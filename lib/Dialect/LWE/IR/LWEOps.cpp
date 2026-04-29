#include "lib/Dialect/LWE/IR/LWEOps.h"

#include <cassert>
#include <optional>

#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"         // from @llvm-project
#include "llvm/include/llvm/Support/ErrorHandling.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"            // from @llvm-project
#include "mlir/include/mlir/IR/MLIRContext.h"         // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"       // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"           // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace lwe {

//===----------------------------------------------------------------------===//
// Op verifiers
//===----------------------------------------------------------------------===//

LogicalResult RMulOp::verify() { return lwe::verifyMulOp(this); }

// Verification for RLWE_EncryptOp
LogicalResult RLWEEncryptOp::verify() {
  Type keyType = getKey().getType();
  auto keyRing =
      llvm::TypeSwitch<Type, mlir::heir::polynomial::RingAttr>(keyType)
          .Case<lwe::LWEPublicKeyType, lwe::LWESecretKeyType>(
              [](auto key) { return key.getRing(); })
          .Default([](Type) {
            llvm_unreachable("impossible by type constraints");
            return nullptr;
          });

  auto outputRing = getOutput().getType().getCiphertextSpace().getRing();
  if (outputRing != keyRing) {
    return emitOpError() << "RLWEEncryptOp input ring do not match. Key ring: "
                         << keyRing
                         << ". Output ciphertext ring: " << outputRing << ".";
  }
  return success();
}

// Verify Encoding and Type match
LogicalResult verifyEncodingAndTypeMatch(mlir::Type type,
                                         mlir::Attribute encoding) {
  // En/Decode Ops only allow IntegerOrFloatLike (-> assert not if)
  assert(getElementTypeOrSelf(type).isIntOrFloat() &&
         "Encoding Ops only allow IntegerOrFloatLike types");

  // Verification conditions for each encoding we have:

  if (isa<FullCRTPackingEncodingAttr>(encoding)) {
    // also supports lists of integers and scalars via replication
    return success(getElementTypeOrSelf(type).isInteger());
  }

  if (isa<InverseCanonicalEncodingAttr>(encoding)) {
    // CKKS-style Encoding should support everything
    // (ints via cast to float/double, scalars via replication)
    return success();
  }

  // This code should never be hit unless we added an encoding and forgot to
  // update this function. Assert(false) for DEBUG, return failure for NDEBUG.
  encoding.dump();
  assert(false && "Encoding not handled in encode/decode verifier.");
  return failure();
}

LogicalResult RLWEEncodeOp::verify() {
  return verifyEncodingAndTypeMatch(getInput().getType(), getEncoding());
}

LogicalResult RLWEDecodeOp::verify() {
  return verifyEncodingAndTypeMatch(getResult().getType(), getEncoding());
}

//===----------------------------------------------------------------------===//
// Op type inference.
//===----------------------------------------------------------------------===//

// LogicalResult RAddOp::inferReturnTypes(
//     MLIRContext* ctx, std::optional<Location>, RAddOp::Adaptor adaptor,
//     SmallVectorImpl<Type>& inferredReturnTypes) {
//   return lwe::inferAddOpReturnTypes(ctx, adaptor, inferredReturnTypes);
// }
//
// LogicalResult RSubOp::inferReturnTypes(
//     MLIRContext* ctx, std::optional<Location>, RSubOp::Adaptor adaptor,
//     SmallVectorImpl<Type>& inferredReturnTypes) {
//   return lwe::inferAddOpReturnTypes(ctx, adaptor, inferredReturnTypes);
// }
//
// LogicalResult RMulOp::inferReturnTypes(
//     MLIRContext* ctx, std::optional<Location>, RMulOp::Adaptor adaptor,
//     SmallVectorImpl<Type>& inferredReturnTypes) {
//   return lwe::inferMulOpReturnTypes(ctx, adaptor, inferredReturnTypes);
// }

}  // namespace lwe
}  // namespace heir
}  // namespace mlir
