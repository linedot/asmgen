from ...registers import (
    reg_tracker,
    asm_data_type as adt,
    adt_triple,
    adt_size,
    adt_is_float,adt_is_int,
    adt_is_signed,adt_is_unsigned,
    asm_index_type as ait,
    data_reg,
    treg,vreg,freg,greg
)
from ..operations import opd3,widening_method,modifier

from ..types.rvv_types import rvv_vreg

from typing import TypeAlias, Callable

class rvv_fma(opd3):

    greg_type : TypeAlias = greg
    freg_type : TypeAlias = freg
    vreg_type : TypeAlias = rvv_vreg
    treg_type : TypeAlias = treg

    def __init__(self,
                 asmwrap : Callable[[str],str]):
        self.asmwrap = asmwrap

    @property
    def widening_method(self) -> widening_method:
        return widening_method.vec_group

    def supported_triples(self) -> list[adt_triple]:
        return [
            adt_triple(a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64),
            adt_triple(a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32),
            adt_triple(a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16),

            adt_triple(a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP64),
            adt_triple(a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP32),

            adt_triple(a_dt=adt.SINT64, b_dt=adt.SINT64, c_dt=adt.SINT64),
            adt_triple(a_dt=adt.SINT32, b_dt=adt.SINT32, c_dt=adt.SINT32),
            adt_triple(a_dt=adt.SINT16, b_dt=adt.SINT16, c_dt=adt.SINT16),
            adt_triple(a_dt=adt.SINT8,  b_dt=adt.SINT8,  c_dt=adt.SINT8),

            adt_triple(a_dt=adt.SINT32, b_dt=adt.SINT32, c_dt=adt.SINT64),
            adt_triple(a_dt=adt.SINT16, b_dt=adt.SINT16, c_dt=adt.SINT32),
            adt_triple(a_dt=adt.SINT8,  b_dt=adt.SINT8,  c_dt=adt.SINT16),

            adt_triple(a_dt=adt.UINT32, b_dt=adt.UINT32, c_dt=adt.UINT64),
            adt_triple(a_dt=adt.UINT16, b_dt=adt.UINT16, c_dt=adt.UINT32),
            adt_triple(a_dt=adt.UINT8,  b_dt=adt.UINT8,  c_dt=adt.UINT16),

            adt_triple(a_dt=adt.SINT32, b_dt=adt.UINT32, c_dt=adt.SINT64),
            adt_triple(a_dt=adt.SINT16, b_dt=adt.UINT16, c_dt=adt.SINT32),
            adt_triple(a_dt=adt.SINT8,  b_dt=adt.UINT8,  c_dt=adt.SINT16),

            adt_triple(a_dt=adt.UINT32, b_dt=adt.SINT32, c_dt=adt.SINT64),
            adt_triple(a_dt=adt.UINT16, b_dt=adt.SINT16, c_dt=adt.SINT32),
            adt_triple(a_dt=adt.UINT8,  b_dt=adt.SINT8,  c_dt=adt.SINT16),
        ]

    def inst_prefix(self, a_dt : adt, b_dt : adt, c_dt : adt) -> str:
        if adt_is_float(a_dt):
            return "vf"
        if adt_is_int(a_dt):
            return "v"

        raise RuntimeError("Unsupported datatype")
    
    def inst_suffix(self, a_dt : adt, b_dt : adt, c_dt : adt) -> str:
        if adt_is_float(a_dt) and adt_is_float(b_dt) and adt_is_float(c_dt):
            return ""
        if adt_is_signed(a_dt) and adt_is_signed(b_dt) and adt_is_signed(c_dt):
            return ""
        if adt_is_unsigned(a_dt) and adt_is_unsigned(b_dt) and adt_is_unsigned(c_dt):
            return "u"
        if adt_is_signed(a_dt) and adt_is_unsigned(b_dt):
            return "su"
        if adt_is_unsigned(a_dt) and adt_is_signed(b_dt):
            return "us"

        raise RuntimeError("Unsupported datatype")

    def __call__(self, adreg : vreg_type, bdreg : vreg_type, cdreg : vreg_type,
                 a_dt : adt, b_dt : adt, c_dt : adt,
                 modifiers : set[modifier] = set(),
                 **kwargs) -> str:
        self.check_triple(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)
        #TODO: better system for checks
        if modifier.regidx in modifiers:
            raise ValueError("RVV has no regidx form")
        if modifier.idx in modifiers:
            raise ValueError("RVV has no idx form")
        if modifier.part in modifiers:
            raise ValueError("RVV has no partial instructions (using vgroups instead)")
        if (a_dt != b_dt):
            raise ValueError("A and B must have same type")
        if (adt_size(a_dt) > adt_size(c_dt)):
            raise ValueError("C type can't have smaller size than A/B type")
        if (adt_is_float(c_dt) and adt_is_int(a_dt)) or\
           (adt_is_float(a_dt) and adt_is_int(c_dt)):
               raise ValueError("Accumulator and multiplicands must be both either fp or int types")
        if (adt_size(a_dt) < adt_size(c_dt)):
            if adt_is_int(c_dt) and modifiers.np in modifiers:
               raise ValueError("RVV has no np form for widening integer operation")
        if (adt_size(c_dt)//adt_size(a_dt)) > 2:
               raise ValueError("RVV only supports 2*SEW widening")


        pref = self.inst_prefix(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)
        mix_pref = "w" if adt_size(c_dt)>adt_size(a_dt) else ""
        base_inst = "nmsac" if modifier.np in modifiers else "macc"
        suf = self.inst_suffix(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)
        form_suf = "vf" if modifier.vf in modifiers else "vv"

        inst = pref+mix_pref+base_inst+suf+"."+form_suf

        inst_str = f"{inst} {cdreg},{adreg},{bdreg}"

        return self.asmwrap(inst_str)
