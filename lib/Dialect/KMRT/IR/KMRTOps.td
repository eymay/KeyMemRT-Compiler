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
    and returns it as a rot_key type. The returned rotation key will have
    the same index as the input parameter.
    
    This operation should typically be preceded by a prefetch_key operation
    with the same index, and followed by a clear_key operation on the result.
  }];
  
  let arguments = (ins I64:$index);
  let results = (outs KMRT_RotKeyType:$rot_key);
  let assemblyFormat = "$index attr-dict `:` type($rot_key)";
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
