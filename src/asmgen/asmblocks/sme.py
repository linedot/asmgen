from asmgen.asmblocks.noarch import reg_tracker
from asmgen.registers import ( 
    asm_data_type as adt,
    asm_index_type as ait,
    adt_triple,
    data_reg,
    treg,vreg,freg,greg,
    adt_size
)
from asmgen.asmblocks.sve import sve,sve_vreg

from asmgen.asmblocks.operations import opd3,widening_method,modifier

from typing import TypeAlias,Callable

class sme_treg(treg):
    def __init__(self, reg_idx : int):
        self.reg_str = f"za{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str

class sme_fmopa(opd3):

    greg_type : TypeAlias = greg
    freg_type : TypeAlias = freg
    vreg_type : TypeAlias = sve_vreg
    treg_type : TypeAlias = sme_treg

    def __init__(self,
                 asmwrap : Callable[[str],str],
                 dt_suffixes : dict[adt,str]):
        self.asmwrap = asmwrap
        self.dt_suffixes = dt_suffixes

        self.NIE_MESSAGE = "Not supported in SME"

    @property
    def widening_method(self) -> widening_method:
        return widening_method.dot_neighbours

    def supported_triples(self) -> list[adt_triple]:
        return [
            adt_triple(a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64),
            adt_triple(a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32),
            adt_triple(a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16),

            adt_triple(a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP32),

            adt_triple(a_dt=adt.FP8E5M2, b_dt=adt.FP8E5M2, c_dt=adt.FP32),
            adt_triple(a_dt=adt.FP8E5M2, b_dt=adt.FP8E5M2, c_dt=adt.FP16),
            adt_triple(a_dt=adt.FP8E4M3, b_dt=adt.FP8E4M3, c_dt=adt.FP32),
            adt_triple(a_dt=adt.FP8E4M3, b_dt=adt.FP8E4M3, c_dt=adt.FP16),

            adt_triple(a_dt=adt.FP8E5M2, b_dt=adt.FP8E5M2, c_dt=adt.FP32),
            adt_triple(a_dt=adt.FP8E5M2, b_dt=adt.FP8E5M2, c_dt=adt.FP16),
            adt_triple(a_dt=adt.FP8E4M3, b_dt=adt.FP8E4M3, c_dt=adt.FP32),
            adt_triple(a_dt=adt.FP8E4M3, b_dt=adt.FP8E4M3, c_dt=adt.FP16),

            adt_triple(a_dt=adt.SINT16, b_dt=adt.SINT16, c_dt=adt.SINT64),
            adt_triple(a_dt=adt.SINT16, b_dt=adt.SINT16, c_dt=adt.SINT32),
            adt_triple(a_dt=adt.SINT8,  b_dt=adt.SINT8,  c_dt=adt.SINT32),

            adt_triple(a_dt=adt.UINT16, b_dt=adt.UINT16, c_dt=adt.UINT64),
            adt_triple(a_dt=adt.UINT16, b_dt=adt.UINT16, c_dt=adt.UINT32),
            adt_triple(a_dt=adt.UINT8,  b_dt=adt.UINT8,  c_dt=adt.UINT32),

            adt_triple(a_dt=adt.SINT16, b_dt=adt.UINT16, c_dt=adt.SINT64),
            adt_triple(a_dt=adt.SINT16, b_dt=adt.UINT16, c_dt=adt.SINT32),
            adt_triple(a_dt=adt.SINT8,  b_dt=adt.UINT8,  c_dt=adt.SINT32),

            adt_triple(a_dt=adt.UINT16, b_dt=adt.SINT16, c_dt=adt.SINT64),
            adt_triple(a_dt=adt.UINT16, b_dt=adt.SINT16, c_dt=adt.SINT32),
            adt_triple(a_dt=adt.UINT8,  b_dt=adt.SINT8,  c_dt=adt.SINT32),
        ]

    def mopx_inst_str(self, a_dt : adt, b_dt : adt, suf : str) -> str:
        if a_dt in [adt.FP8E5M2, adt.FP8E4M3, adt.FP16, adt.FP32, adt.FP64]:
            return f"fmop{suf}"
        if a_dt in [adt.BF16]:
            return f"bfmop{suf}"
        if a_dt in [adt.UINT8, adt.UINT16] and b_dt in [adt.UINT8, adt.UINT16]:
            return f"umop{suf}"
        if a_dt in [adt.SINT8, adt.SINT16] and b_dt in [adt.SINT8, adt.SINT16]:
            return f"smop{suf}"
        if a_dt in [adt.SINT8, adt.SINT16] and b_dt in [adt.UINT8, adt.UINT16]:
            return f"sumop{suf}"
        if a_dt in [adt.UINT8, adt.UINT16] and b_dt in [adt.SINT8, adt.SINT16]:
            return f"usmop{suf}"

        raise ValueError("Unsupported datatypes")

    def __call__(self, adreg : vreg_type, bdreg : vreg_type, cdreg : treg_type,
                 a_dt : adt, b_dt : adt, c_dt : adt,
                 modifiers : set[modifier] = set(),
                 **kwargs) -> str:
        self.check_triple(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)

        if modifiers.intersection(
                set([modifier.idx, modifier.part, modifier.vf])):
            raise ValueError("unsupported modifiers for SME")
        if (a_dt != b_dt):
            raise ValueError("A and B must have same type")
        if (adt_size(a_dt) > adt_size(c_dt)):
            raise ValueError("C type can't have smaller size than A/B type")
        fp_types = [adt.FP64, adt.FP32, adt.FP16, adt.FP8E4M3, adt.FP8E5M2]
        int_types = [adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8,
                     adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8]
        if (c_dt in fp_types and a_dt in int_types) or\
           (c_dt in int_types and a_dt in fp_types):
               raise ValueError("Accumulator and multiplicands must be both either fp or int types")
        valid_c_types = [adt.FP64, adt.FP32, adt.FP16,
                         adt.UINT64, adt.UINT32, adt.UINT16,
                         adt.SINT64, adt.SINT32, adt.SINT16]
        if c_dt not in valid_c_types:
            raise ValueError(f"C type must be one of [{','.join(valid_c_types)}]")


        suf = "s" if modifier.np in modifiers else "a"
        inst = self.mopx_inst_str(a_dt=a_dt, b_dt=b_dt, suf=suf)
        narrow_suf = self.dt_suffixes[a_dt]
        wide_suf = self.dt_suffixes[c_dt]
        return self.asmwrap(f"{inst} {cdreg}.{wide_suf},p0/m,p0/m,{adreg}.{narrow_suf},{bdreg}.{narrow_suf}")

class sme(sve):
    greg_type : TypeAlias = greg
    freg_type : TypeAlias = freg
    vreg_type : TypeAlias = sve_vreg
    treg_type : TypeAlias = sme_treg

    def __init__(self):
        super(sme, self).__init__()
        self.fopa = sme_fmopa(asmwrap=self.asmwrap,
                              dt_suffixes=self.dt_suffixes)

    @property
    def c_simd_size_function(self):
        result  = "size_t get_simd_size() {\n"
        result += "    size_t byte_size = 0;\n"
        result += "    __asm__ volatile(\n"
        result += "        "+self.asmwrap("smstart")
        result += "        "+self.asmwrap("mov %[byte_size],#0")
        result += "        "+self.asmwrap("incb %[byte_size]")
        result += "        "+self.asmwrap("smstop")
        result += "    : [byte_size] \"=r\" (byte_size)\n"
        result += "    :\n"
        result += "    :\n"
        result += "    );\n"
        result += "    return byte_size;\n"
        result += "}"
        return result

    @property
    def max_tregs(self, dt : adt) -> int:
        return adt_size(dt)

    def treg(self, reg_idx : int) -> treg_type:
        return sme_treg(reg_idx)

    def zero_treg(self, atreg : sme_treg, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        return self.asmwrap("zero {atreg}.{suf}")

    def store_tile(self, areg : greg_type, ignored_offset : int,
                   atreg : treg_type, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        raise NotImplementedError("Not implemented yet")
