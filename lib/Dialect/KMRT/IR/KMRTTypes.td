#ifndef LIB_DIALECT_KMRT_IR_KMRTTYPES_TD_
#define LIB_DIALECT_KMRT_IR_KMRTTYPES_TD_

include "lib/Dialect/KMRT/IR/KMRTDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpAsmInterface.td"

class KMRT_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<KMRT_Dialect, name, traits # [OpAsmTypeInterface]> {
  let mnemonic = typeMnemonic;
  let assemblyFormat = "`<` struct(params) `>`";

  string asmName = ?;
  let extraClassDeclaration = [{
    // OpAsmTypeInterface method
    void getAsmName(::mlir::OpAsmSetNameFn setNameFn) const {
      setNameFn("}] # asmName # [{");
    }

    ::mlir::OpAsmDialectInterface::AliasResult getAlias(::llvm::raw_ostream &os) const {
      os << "}] # asmName # [{";
      return ::mlir::OpAsmDialectInterface::AliasResult::FinalAlias;
    }
  }];
}

def KMRT_RotKeyType : KMRT_Type<"RotKey", "rot_key"> {
  let summary = "A rotation key type with an index parameter";
  let description = [{
    The RotKeyType represents rotation keys used in the KeyMemRT system.
    Each rotation key has an associated rotation_index that identifies the specific
    rotation amount or key identifier.
  }];

  let parameters = (ins "int64_t":$rotation_index);
  let asmName = "rk";
}

#endif // LIB_DIALECT_KMRT_IR_KMRTTYPES_TD_
