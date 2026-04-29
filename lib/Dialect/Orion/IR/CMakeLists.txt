add_heir_dialect(Orion orion)

add_mlir_dialect_library(HEIROrion
    OrionDialect.cpp

    DEPENDS
    HEIROrionIncGen
)
