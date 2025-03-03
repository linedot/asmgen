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

from ..types.neon_types import neon_vreg

from typing import TypeAlias, Callable

class neon_fmul(opd3):

    greg_type : TypeAlias = greg
    freg_type : TypeAlias = freg
    vreg_type : TypeAlias = neon_vreg
    treg_type : TypeAlias = treg

    def __init__(self,
                 asmwrap : Callable[[str],str],
                 dt_suffixes : dict[adt,str],
                 dt_idxsuffixes : dict[adt,str],
                 ):
        self.asmwrap = asmwrap
        self.dt_suffixes = dt_suffixes
        self.dt_idxsuffixes = dt_idxsuffixes

    @property
    def widening_method(self) -> widening_method:
        return widening_method.split_instructions

    def supported_triples(self) -> list[adt_triple]:
        return [
            adt_triple(a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64),
            adt_triple(a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32),
            adt_triple(a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16),

            adt_triple(a_dt=adt.SINT32, b_dt=adt.SINT32, c_dt=adt.SINT32),
            adt_triple(a_dt=adt.SINT16, b_dt=adt.SINT16, c_dt=adt.SINT16),
            adt_triple(a_dt=adt.SINT8,  b_dt=adt.SINT8,  c_dt=adt.SINT8),

            # TODO: more types
        ]

    def mlx_inst_str(self, a_dt : adt, b_dt : adt, c_dt : adt) -> str:
        if a_dt in [adt.FP8E5M2, adt.FP8E4M3, adt.FP16, adt.FP32, adt.FP64]:
            return f"fmul"
        if a_dt in [adt.BF16]:
            return f"bfmul"
        if a_dt in [adt.SINT8, adt.SINT16, adt.SINT32, adt.SINT64] and (a_dt == c_dt):
            return f"mul"
        if a_dt in [adt.UINT8, adt.UINT16] and b_dt in [adt.UINT8, adt.UINT16]:
            return f"umul"
        if a_dt in [adt.SINT8, adt.SINT16] and b_dt in [adt.SINT8, adt.SINT16]:
            return f"smul"

        raise RuntimeError("Unsupported datatype")

    def partial_inst_suffix(self, ways : int, part : int):
        if ways.bit_count() != 1:
            raise ValueError(f"ways={ways} is not a power of 2")
        suf = 'l'*(ways.bit_length()-1)

        suf += f"{part:b>{ways.bit_length()-1}b}".replace('1','t').replace('0','b')

        return suf


    def __call__(self, adreg : vreg_type, bdreg : vreg_type, cdreg : vreg_type,
                 a_dt : adt, b_dt : adt, c_dt : adt,
                 modifiers : set[modifier] = [],
                 **kwargs) -> str:
        self.check_triple(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)

        if modifier.np in modifiers:
            raise ValueError("NEON FMUL has no np form")

        if modifier.vf in modifiers:
            raise ValueError("NEON has no vf form")

        if modifier.regidx in modifiers:
            raise ValueError("NEON has no regidx form")

        if (a_dt != b_dt):
            raise ValueError("A and B must have same type")

        if (adt_size(a_dt) > adt_size(c_dt)):
            raise ValueError("C type can't have smaller size than A/B type")

        if (adt_is_float(c_dt) and adt_is_int(a_dt)) or\
           (adt_is_float(a_dt) and adt_is_int(c_dt)):
               raise ValueError("Accumulator and multiplicands must be both either fp or int types")

        valid_c_types = [adt.FP64, adt.FP32, adt.FP16,
                         adt.UINT64, adt.UINT32, adt.UINT16,
                         adt.SINT64, adt.SINT32, adt.SINT16]
        if c_dt not in valid_c_types:
            raise ValueError(f"C type must be one of [{','.join(valid_c_types)}]")

        if adt_is_float(c_dt) and (adt_size(c_dt) > adt_size(a_dt)):
            raise ValueError(f"NEON FMUL has no widening FP version")

        if (adt_size(c_dt)//adt_size(a_dt) > 2):
            raise ValueError(f"NEON FMUL has only 2xways widening")

        if c_dt in [adt.UINT64, adt.UINT32, adt.UINT16]:
            if adt_size(a_dt) == adt_size(c_dt):
                raise ValueError(f"only widening variants exist for unsigned integer types")

        if (adt_size(a_dt) < adt_size(c_dt)):
            if (not modifier.part in modifiers) or ('part' not in kwargs):
                raise ValueError("NEON requires 'part' modifier and argument for widening operations")
            part = kwargs['part']

        if modifier.idx in modifiers:
            if 'idx' not in kwargs:
                raise ValueError("'idx' modifier specified, but not 'idx' parameter")
            idx = kwargs['idx']



        inst = self.mlx_inst_str(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)
        if modifier.part in modifiers:
            ways = adt_size(c_dt)//adt_size(a_dt)
            inst += self.partial_inst_suffix(ways=ways, part=part)

        narrow_suf = self.dt_suffixes[a_dt]
        wide_suf = self.dt_suffixes[c_dt]
        inst_str = f"{inst} {cdreg}.{wide_suf},{adreg}.{narrow_suf},"
        if modifier.idx in modifiers:
            b_suf = self.dt_idxsuffixes[b_dt]
            inst_str += f"{bdreg}.{b_suf}[{idx}]"
        else:
            b_suf = narrow_suf
            inst_str += f"{bdreg}.{b_suf}"

        return self.asmwrap(inst_str)
