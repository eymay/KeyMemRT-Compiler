#ifndef LIB_DIALECT_KMRT_IR_KMRTOPS_TD_
#define LIB_DIALECT_KMRT_IR_KMRTOPS_TD_

include "lib/Dialect/KMRT/IR/KMRTDialect.td"
include "lib/Dialect/KMRT/IR/KMRTTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def KMRT_PrefetchKeyOp : Op<KMRT_Dialect, "prefetch_key", [
    MemoryEffects<[MemWrite]>  // Has side effects - modifies cache/memory state
  ]> {
  let summary = "Prefetch a rotation key by index";
  let description = [{
    The prefetch_key operation initiates prefetching of a rotation key
    identified by the given index. This operation has side effects as it
    may trigger memory operations or cache warming.
    
    This operation should typically be followed by a load_key operation
    with the same index.
  }];
  
  let arguments = (ins I64:$index);
  let results = (outs);
  let assemblyFormat = "$index attr-dict";
}

def KMRT_LoadKeyOp : Op<KMRT_Dialect, "load_key", [
    MemoryEffects<[MemRead]>  // Reads from memory/cache
  ]> {
  let summary = "Load a rotation key by index";
  let description = [{
    The load_key operation loads a rotation key identified by the given index
    and returns it as a rot_key type.

    The index operand can be:
    - A constant value (e.g., from arith.constant) → produces static rot_key type
    - An affine induction variable or computed index → produces dynamic rot_key type

    The result type indicates whether this is a static or dynamic rotation:
    - !kmrt.rot_key<rotation_index = N> for static constant N
    - !kmrt.rot_key for dynamic (runtime-determined) indices

    For dynamic rotation keys, analysis passes examine the index operand's
    def-use chain to determine possible values (e.g., from affine loop bounds).

    This operation should typically be preceded by a prefetch_key operation
    with the same index, and followed by a clear_key operation on the result.

    Examples:
      // Static constant index
      %c5 = arith.constant 5 : i64
      %rk = kmrt.load_key %c5 : !kmrt.rot_key<rotation_index = 5>

      // Dynamic index from affine loop
      affine.for %iv = 1 to 16 {
        %iv_i64 = arith.index_cast %iv : index to i64
        %rk = kmrt.load_key %iv_i64 : !kmrt.rot_key
      }

      // Can also accept index type directly
      affine.for %iv = 1 to 16 {
        %rk = kmrt.load_key %iv : !kmrt.rot_key
      }
  }];

  let arguments = (ins AnyTypeOf<[Index, I64]>:$index);
  let results = (outs KMRT_RotKeyType:$rot_key);
  let assemblyFormat = "$index attr-dict `:` type($index) `->` type($rot_key)";
}

def KMRT_ClearKeyOp : Op<KMRT_Dialect, "clear_key", [
    MemoryEffects<[MemWrite]>  // Has side effects - modifies memory/cache state
  ]> {
  let summary = "Clear a rotation key from memory";
  let description = [{
    The clear_key operation removes a rotation key from memory or cache.
    This operation consumes the rotation key and has side effects.
    
    This operation should typically be preceded by a load_key operation
    that produced the rotation key being cleared.
  }];
  
  let arguments = (ins KMRT_RotKeyType:$rot_key);
  let results = (outs);
  let assemblyFormat = "$rot_key attr-dict `:` type($rot_key)";
}

def KMRT_RotationOp : Op<KMRT_Dialect, "rotation", [Pure, AllTypesMatch<["ciphertext", "result"]>]> {
  let summary = "Perform rotation using a loaded rotation key";
  let description = [{
    The rotation operation performs a cyclic rotation on a ciphertext using
    a previously loaded rotation key. This is the KMRT equivalent of 
    OpenFHE's rotation operations, but explicitly uses a rotation key value.
  }];
  
  let arguments = (ins AnyType:$ciphertext, KMRT_RotKeyType:$rot_key);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$ciphertext `,` $rot_key attr-dict `:` type($ciphertext) `,` type($rot_key) `->` type($result)";
}

#endif  // LIB_DIALECT_KMRT_IR_KMRTOPS_TD_
