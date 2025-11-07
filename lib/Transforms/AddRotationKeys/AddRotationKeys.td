#ifndef LIB_TRANSFORMS_ADDROTATIONKEYS_ADDROTATIONKEYS_TD_
#define LIB_TRANSFORMS_ADDROTATIONKEYS_ADDROTATIONKEYS_TD_

include "mlir/Pass/PassBase.td"


def AddRotationKeys : Pass<"add-rotation-keys"> {
  let summary = "Add or replace rotation keys with specified indices";
  let description = [{
    This pass manages the rotation keys in GenRotKeyOp and GenRotKeyDepthOp
    operations by setting rotation indices provided via command line.

    The pass takes a comma-separated list of rotation indices and either:
    - Overwrites the existing rotation indices (default: --overwrite=true)
    - Merges them with existing rotation indices (--overwrite=false)

    When overwrite=false, it performs set union with existing keys and warns
    if there are keys that are not part of the provided list.

    Example usage:
    --add-rotation-keys="1,2,4,8,16,32,64,128,256,512,1024,2048,4096"
    --add-rotation-keys="1,2,4,8" --overwrite=false  (merge mode)
  }];

  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];

  let options = [
    Option<"rotationIndices", "rotation-indices", "std::string",
           /*default=*/"\"\"", 
           "Comma-separated list of rotation indices to add">,
    Option<"overwrite", "overwrite", "bool",
           /*default=*/"true",
           "Overwrite the indices found in previous passes">
  ];
}


#endif  // LIB_TRANSFORMS_ADDROTATIONKEYS_ADDROTATIONKEYS_TD_
