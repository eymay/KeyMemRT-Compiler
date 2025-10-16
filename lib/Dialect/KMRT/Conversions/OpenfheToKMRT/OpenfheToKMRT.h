#ifndef LIB_DIALECT_KMRT_CONVERSIONS_OPENFHETOKMRTT_OPENFHETOKMRTT_H_
#define LIB_DIALECT_KMRT_CONVERSIONS_OPENFHETOKMRTT_OPENFHETOKMRTT_H_

#include "mlir/include/mlir/Pass/Pass.h"

namespace mlir {
namespace keymemrt {
namespace kmrt {

#define GEN_PASS_DECL
#include "lib/Dialect/KMRT/Conversions/OpenfheToKMRT/OpenfheToKMRT.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Dialect/KMRT/Conversions/OpenfheToKMRT/OpenfheToKMRT.h.inc"

} // namespace kmrt
} // namespace keymemrt
} // namespace mlir

#endif // LIB_DIALECT_KMRT_CONVERSIONS_OPENFHETOKMRTT_OPENFHETOKMRTT_H_
