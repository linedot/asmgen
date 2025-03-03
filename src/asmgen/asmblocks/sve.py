from .aarch64 import aarch64

from ..registers import (
    reg_tracker,
    asm_data_type as adt,
    adt_triple,
    adt_size,
    asm_index_type as ait,
    data_reg,
    treg,vreg,freg,greg
)
from asmgen.asmblocks.operations import opd3,widening_method,modifier

from .types.sve_types import sve_vreg
from .sve_opd3 import sve_fma,sve_fmul

from typing import TypeAlias, Callable


class sve(aarch64):

    greg_type : TypeAlias = greg
    freg_type : TypeAlias = freg
    vreg_type : TypeAlias = sve_vreg
    treg_type : TypeAlias = treg

    dt_suffixes = {
            adt.DOUBLE  : "d",
            adt.UINT64  : "d",
            adt.SINT64  : "d",
            adt.SINGLE  : "s",
            adt.UINT32  : "s",
            adt.SINT32  : "s",
            adt.HALF    : "h",
            adt.UINT16  : "h",
            adt.SINT16  : "h",
            adt.FP8E5M2 : "b",
            adt.FP8E4M3 : "b",
            adt.UINT8   : "b",
            adt.SINT8   : "b",
            }
    dt_mnem_suffixes = {
            adt.DOUBLE : "d",
            adt.SINGLE : "w",
            }

    def __init__(self):
        super(sve, self).__init__()
        self.fma = sve_fma(asmwrap=self.asmwrap,
                           dt_suffixes=self.dt_suffixes)
        self.fmul = sve_fmul(asmwrap=self.asmwrap,
                           dt_suffixes=self.dt_suffixes)

    def get_req_flags(self):
        return ['sve']

    def supportedby_cpuinfo(self, cpuinfo):
         req_flags = self.get_req_flags()
         supported = True
         for r in req_flags:
             if -1 == cpuinfo.find(r):
                 supported = False
                 break
         return supported

    def isaquirks(self, rt : reg_tracker, dt : adt):
        asmblock = self.ptrue(self.preg(0), dt)
        return asmblock

    def jvzero(self, vreg1 : vreg, freg : freg,
               vreg2 : vreg,
               greg : greg, label : str,
               datatype : adt):
        suf = self.dt_suffixes[datatype]
        asmblock  = self.asmwrap(f"fcmne p1.d,{vreg}.{suf},p0/z,#0,0")
        asmblock += self.asmwrap(f"ptest p0, p1.b")
        asmblock += self.asmwrap(f"b.any .{label}")
        return asmblock


    @property
    def is_vla(self):
        return True

    @property
    def max_vregs(self):
        return 32

    @property
    def simd_size(self):
        return 1

    def indexable_elements(self, datatype):
        # 128 bits are indexable
        return 16/datatype.value

    @property
    def c_simd_size_function(self):
        result  = "size_t get_simd_size() {\n"
        result += "    size_t byte_size = 0;\n"
        result += "    __asm__ volatile(\n"
        result += "        "+self.asmwrap("mov %[byte_size],#0")
        result += "        "+self.asmwrap("incb %[byte_size]")
        result += "    : [byte_size] \"=r\" (byte_size)\n"
        result += "    :\n"
        result += "    :\n"
        result += "    );\n"
        result += "    return byte_size;\n"
        result += "}"
        return result

    def fmul(self, avreg : vreg_type, bvreg : vreg_type, cvreg : vreg_type, a_dt : adt, b_dt : adt, c_dt : adt):
        suf = self.dt_suffixes[datatype]
        return self.asmwrap(f"fmul {dst}.{suf},p0/m,{a}.{suf},{b}.{suf}")

    def add_greg_voff(self, reg, offset, datatype):
        return self.asmwrap(f"incb {reg}, ALL, MUL #{offset}")

    def zero_vreg(self,i,datatype):
        suf = self.dt_suffixes[datatype]
        return self.asmwrap(f"dup {i}.{suf},#0")

    def vreg(self, reg_idx : int) -> sve_vreg:
        return sve_vreg(reg_idx)

    def min_load_immoff(self,datatype):
        return 0

    def max_load_immoff(self,datatype):
        return 252

    @property
    def min_load_voff(self):
        return -8

    @property
    def max_load_voff(self):
        return 7

    def load_vector(self, a, ignored_offset, v, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        suf = self.dt_suffixes[datatype]
        msuf = self.dt_mnem_suffixes[datatype]
        return self.asmwrap(f"ld1{msuf} {v}.{suf}, p0/z, [{a}]")

    def load_vector_voff(self, a, voffset, v, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        suf = self.dt_suffixes[datatype]
        msuf = self.dt_mnem_suffixes[datatype]
        return self.asmwrap(f"ld1{msuf} {v}.{suf}, p0/z, [{a}, #{voffset}, MUL VL]")

    def load_vector_dist1(self, a, ignored_offset, v, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        suf = self.dt_suffixes[datatype]
        msuf = self.dt_mnem_suffixes[datatype]
        return self.asmwrap(f"ld1r{msuf} {v}.{suf}, p0/z, [{a}]")

    def load_vector_dist1_boff(self, a, offset, v, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        suf = self.dt_suffixes[datatype]
        msuf = self.dt_mnem_suffixes[datatype]
        return self.asmwrap(f"ld1r{msuf} {v}.{suf}, p0/z, [{a}, #{offset}]")

    def load_vector_dist1_inc(self, a, ignored_offset, v, datatype):
        raise NotImplementedError("SVE doesn't have a post-index ld1r{suf}, use load_vector_dist1_boff instead")

    def store_vector_voff(self, a, voffset, v, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        suf = self.dt_suffixes[datatype]
        msuf = self.dt_mnem_suffixes[datatype]
        address = f"[{a}, #{voffset}, MUL VL]"
        if 0 == voffset:
            address = f"[{a}]"
        return self.asmwrap(f"st1{msuf} {{{v}.{suf}}}, p0, {address}")

    def store_vector(self, a, voffset, v, datatype):
        return self.store_vector_voff(a, 0, v, datatype)

    # SVE-specific
    def preg(self, i):
        return f"p{i}"

    def ptrue(self, reg, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        suf = self.dt_suffixes[datatype]
        return self.asmwrap(f"ptrue {reg}.{suf}")


    def load_vector_immstride(self, areg : greg_type, byte_stride : int,
                    vreg : vreg_type, datatype : adt):
        raise NotImplementedError("SVE has no load with immediate stride")

    def load_vector_gregstride(self, areg : greg_type, sreg : greg_type,
                    vreg : vreg_type, datatype : adt):
        raise NotImplementedError("SVE has no load with scalar register stride")

    def load_vector_gather(self, areg : greg_type, offvreg : vreg_type,
                           vreg : vreg_type, datatype : adt,
                           indextype : ait):
        suf = self.dt_suffixes[datatype]
        msuf = self.dt_mnem_suffixes[datatype]
        return self.asmwrap(f"ld1{msuf}.v {vreg}.{suf}, p0/z,[{areg}, {offvreg}]")

    def store_vector_immstride(self, areg : greg_type, byte_stride : int,
                    vreg : vreg_type, datatype : adt):
        raise NotImplementedError("RVV has no store with immediate stride")

    def store_vector_gregstride(self, areg : greg_type, sreg : greg_type,
                    vreg : vreg_type, datatype : adt):
        raise NotImplementedError("SVE has no store with scalar register stride")

    def store_vector_scatter(self, areg : greg_type, offvreg : vreg_type,
                             vreg : vreg_type, datatype : adt,
                             indextype : ait):
        suf = self.dt_suffixes[datatype]
        msuf = self.dt_mnem_suffixes[datatype]
        return self.asmwrap(f"st1{msuf}.v {vreg}.{suf}, p0, [{areg}, {offvreg}]")


    # Unsupported functionality:
    def max_tregs(self, dt : adt):
        return 0

    def treg(self, reg_idx : int):
        raise NotImplementedError("SVE has no tiles, use SME")

    def zero_treg(self, treg : treg_type, datatype : adt):
        raise NotImplementedError("SVE has no tiles, use SME")

    def store_tile(self, areg : greg_type,
                   ignored_offset : int,
                   treg : treg_type,
                   datatype : adt):
        raise NotImplementedError("SVE has no tiles, use SME")
