from ...registers import (
    reg_tracker,
    asm_data_type as adt,
    adt_triple,
    adt_size,
    adt_is_float,adt_is_int,
    asm_index_type as ait,
    data_reg,
    treg,vreg,freg,greg
)
from ..operations import opd3,widening_method,modifier

from ..types.avx_types import prefix_if_raw_reg,x86_greg

from typing import TypeAlias, Callable

class avx_fmul(opd3):

    greg_type : TypeAlias = x86_greg
    freg_type : TypeAlias = freg
    vreg_type : TypeAlias = vreg
    treg_type : TypeAlias = treg

    def __init__(self,
                 asmwrap : Callable[[str],str],
                 dt_suffixes : dict[adt,str],
                 it_suffixes : dict[ait,str],
                 has_fp16 : bool = False
                 ):
        self.asmwrap = asmwrap
        self.dt_suffixes = dt_suffixes
        self.it_suffixes = it_suffixes
        self.has_fp16 = has_fp16

    @property
    def widening_method(self) -> widening_method:
        raise NotImplementedError("AVX mixed precision not implemented")

    def supported_triples(self) -> list[adt_triple]:
        supported_list = [
            adt_triple(a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64),
            adt_triple(a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32),
        
            #TODO: more types
        ]
        if self.has_fp16:
            supported_list.append(
                    adt_triple(a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16))
        return supported_list

        raise RuntimeError("Unsupported datatype")


    def __call__(self, adreg : vreg_type, bdreg : vreg_type, cdreg : vreg_type,
                 a_dt : adt, b_dt : adt, c_dt : adt,
                 modifiers : set[modifier] = set(),
                 **kwargs) -> str:
        self.check_triple(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)
        #TODO: better system for checks
        if modifier.np in modifiers:
            raise ValueError("AVX FMUL has no np form")
        if modifier.vf in modifiers:
            raise ValueError("AVX has no vf form")
        if modifier.regidx in modifiers:
            raise ValueError("AVX has no regidx form")
        if modifier.idx in modifiers:
            raise ValueError("AVX has no idx form")
        if modifier.part in modifiers:
            raise ValueError("AVX has no partial instructions")
        if (a_dt != b_dt) or (a_dt != c_dt):
            raise ValueError("A,B and C must have same type")

        suf = 'p'+self.dt_suffixes[c_dt]
        pa = prefix_if_raw_reg(adreg)
        pb = prefix_if_raw_reg(bdreg)
        pc = prefix_if_raw_reg(cdreg)
        return self.asmwrap(f"vmul{suf} {pa},{pb},{pc}")
