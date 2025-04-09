"""
C++ data type names
"""
from asmgen.asmblocks.noarch import asm_data_type

c_data_types = {
        asm_data_type.DOUBLE   : "double",
        asm_data_type.UINT64   : "uint64_t",
        asm_data_type.SINT64   : "int64_t",
        asm_data_type.SINGLE   : "float",
        asm_data_type.UINT32   : "uint32_t",
        asm_data_type.SINT32   : "int32_t",
        asm_data_type.HALF     : "uint16_t",
        asm_data_type.UINT16   : "uint16_t",
        asm_data_type.SINT16   : "int16_t",
        asm_data_type.FP8E4M3  : "uint8_t",
        asm_data_type.FP8E5M2  : "uint8_t",
        asm_data_type.UINT8    : "uint8_t",
        asm_data_type.SINT8    : "int8_t"
        }
