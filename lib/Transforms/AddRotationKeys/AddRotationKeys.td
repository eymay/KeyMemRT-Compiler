#ifndef LIB_TRANSFORMS_ADDROTATIONKEYS_ADDROTATIONKEYS_TD_
#define LIB_TRANSFORMS_ADDROTATIONKEYS_ADDROTATIONKEYS_TD_

include "mlir/Pass/PassBase.td"


def AddRotationKeys : Pass<"add-rotation-keys"> {
  let summary = "Extend rotation keys with additional indices";
  let description = [{
    This pass extends the rotation keys in GenRotKeyOp operations by adding
    additional rotation indices provided via command line. It performs set 
    merging with existing keys and warns if there are keys that are not 
    part of the provided list.

    The pass takes a comma-separated list of rotation indices and merges
    them with the existing rotation indices in the IR. If there are existing
    keys that are not in the provided list, it will issue a warning.

    Example usage:
    --add-rotation-keys="1,2,4,8,16,32,64,128,256,512,1024,2048,4096"
  }];

  let dependentDialects = ["mlir::heir::openfhe::OpenfheDialect"];

  let options = [
    Option<"rotationIndices", "rotation-indices", "std::string",
           /*default=*/"\"\"", 
           "Comma-separated list of rotation indices to add">
  ];
}


#endif  // LIB_TRANSFORMS_ADDROTATIONKEYS_ADDROTATIONKEYS_TD_
