from ..registers import (
    reg_tracker,
    asm_data_type as adt,
    adt_triple,
    adt_size,
    asm_index_type as ait,
    data_reg,
    treg,vreg,freg,greg
)
from .aarch64 import aarch64

from .types.neon_types import neon_vreg
from .neon_opd3 import neon_fma,neon_fmul

from typing import TypeAlias

class neon(aarch64):

    greg_type : TypeAlias = greg
    freg_type : TypeAlias = freg
    vreg_type : TypeAlias = neon_vreg
    treg_type : TypeAlias = treg

    dt_suffixes = {
            adt.DOUBLE  : "2d",
            adt.UINT64  : "2d",
            adt.SINT64  : "2d",
            adt.SINGLE  : "4s",
            adt.UINT32  : "4s",
            adt.SINT32  : "4s",
            adt.HALF    : "8h",
            adt.UINT16  : "8h",
            adt.SINT16  : "8h",
            adt.FP8E5M2 : "16b",
            adt.FP8E4M3 : "16b",
            adt.UINT8   : "16b",
            adt.SINT8   : "16b",
            }
    dt_idxsuffixes = {
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

    dt_greg_pfx = {
            adt.DOUBLE  : "x",
            adt.SINGLE  : "w",
            adt.HALF    : "w",
            adt.FP8E5M2 : "w",
            adt.FP8E4M3 : "w",
            }

    def __init__(self):
        super(neon, self).__init__()
        self.fma = neon_fma(asmwrap=self.asmwrap,
                            dt_suffixes=self.dt_suffixes,
                            dt_idxsuffixes=self.dt_idxsuffixes)
        self.fmul = neon_fmul(asmwrap=self.asmwrap,
                              dt_suffixes=self.dt_suffixes,
                              dt_idxsuffixes=self.dt_idxsuffixes)

    def get_req_flags(self):
        return ['asimd']

    def supportedby_cpuinfo(self, cpuinfo : str) -> bool:
         req_flags = self.get_req_flags()
         supported = True
         for r in req_flags:
             if -1 == cpuinfo.find(r):
                 supported = False
                 break
         return supported

    def isaquirks(self, rt : reg_tracker, dt : adt):
        return ""

    def jvzero(self, vreg1 : vreg_type, freg : freg_type,
               vreg2 : vreg_type,
               greg : greg_type, label : str,
               datatype : adt):
        suf = self.dt_suffixes[datatype]
        asmblock  = self.asmwrap(f"fmaxv {freg}, {vreg2}.{suf}")
        asmblock += self.asmwrap(f"fcmp {freg}, #0.0")
        asmblock += self.asmwrap(f"b.eq .{label}")
        return asmblock

    def vreg_to_qreg(self, vreg : neon_vreg):
        return vreg.reg_str.replace('v','q')

    @property
    def is_vla(self):
        return False

    def indexable_elements(self, dt : adt):
        return self.simd_size/adt_size(dt)

    @property
    def max_vregs(self):
        return 32

    @property
    def simd_size(self):
        return 16

    @property
    def c_simd_size_function(self):
        return f"size_t get_simd_size() {{ return {self.simd_size}; }}"

    def add_greg_voff(self, reg : greg_type, offset : int, dt : adt):
        byte_offset = self.simd_size*offset
        return self.asmwrap(f"add {reg},{reg},#{byte_offset}")
        
    def zero_vreg(self, reg : greg_type, dt : adt):
        suf = self.dt_suffixes[dt]
        zeroreg = f"{self.dt_greg_pfx[dt]}zr" 
        return self.asmwrap(f"dup {reg}.{suf},{zeroreg}")

    def vreg(self, reg_idx : int) -> neon_vreg:
        return neon_vreg(reg_idx)

    def qreg(self, i):
        return f"q{i}"

    def min_load_immoff(self, dt : adt):
        return 0

    def max_load_immoff(self, dt : adt):
        return 4095*adt_size(dt)*2

    @property
    def min_load_voff(self):
        return 0

    @property
    def max_load_voff(self):
        # TODO: The immediate can be 0 to 4095, but in add_greg_voff we multiply
        #       with data size. We can  can LSL #12 when adding,
        #       So you could split in large and small part and use 2 instructions
        #       not sure what's up with the actual load. Investigate
        return 4096/8

    def load_vector(self, areg : greg_type, ignored_offset : int, vreg : vreg_type, dt : adt):
        qv = self.vreg_to_qreg(vreg)
        return self.asmwrap(f"ldr {qv}, [{areg}]")

    def load_vector_voff(self, areg : greg_type, voffset : int, vreg : vreg_type, dt : adt):
        qv = self.vreg_to_qreg(vreg)
        return self.asmwrap(f"ldr {qv}, [{areg}, #{voffset*self.simd_size}]")

    def load_vector_dist1(self, areg : greg_type, ignored_offset : int, vreg : vreg_type, dt : adt):
        suf = self.dt_suffixes[dt]
        return self.asmwrap(f"ld1r {{{vreg}.{suf}}}, [{areg}]")

    def load_vector_dist1_boff(self, areg : greg_type, offset : int, v : vreg_type, dt : adt):
        raise NotImplementedError("load_vector_dist1_boff doesn't make sense with NEON, use load_vector_dist1_inc or load_vector_voff + fma_idx instead")
    
    def load_vector_dist1_inc(self, areg : greg_type, ignored_offset : int, vreg : vreg_type, dt : adt):
        suf = self.dt_suffixes[dt]
        return self.asmwrap(f"ld1r {{{vreg}.{suf}}}, [{areg}], #{adt_size(dt)}")

    def store_vector_voff(self, areg : greg_type, voffset : int, vreg : vreg_type, dt : adt):
        qv = self.vreg_to_qreg(vreg)
        return self.asmwrap(f"str {qv}, [{areg}, #{voffset*self.simd_size}]")

    def store_vector(self, areg : greg_type, voffset : int, vreg : vreg_type, dt : adt):
        qv = self.vreg_to_qreg(vreg)
        return self.asmwrap(f"str {qv}, [{areg}]")

    def load_vector_immstride(self, areg : greg_type, byte_stride : int,
                    vreg : vreg_type, datatype : adt):
        raise NotImplementedError("NEON has no load with immediate stride")

    def load_vector_gregstride(self, areg : greg_type, sreg : greg_type,
                    vreg : vreg_type, datatype : adt):
        raise NotImplementedError("NEON has no load with scalar register stride")

    def load_vector_gather(self, areg : greg_type, offvreg : vreg_type,
                           vreg : vreg_type, datatype : adt,
                           indextype : ait):
        raise NotImplementedError("NEON has no load with vector register stride")

    def store_vector_immstride(self, areg : greg_type, byte_stride : int,
                    vreg : vreg_type, datatype : adt):
        raise NotImplementedError("NEON has no store with immediate stride")

    def store_vector_gregstride(self, areg : greg_type, sreg : greg_type,
                    vreg : vreg_type, datatype : adt):
        raise NotImplementedError("NEON has no store with scalar register stride")

    def store_vector_scatter(self, areg : greg_type, offvreg : vreg_type,
                             vreg : vreg_type, datatype : adt,
                             indextype : ait):
        raise NotImplementedError("NEON has no store with vector register stride")

    # Unsupported functionality:
    def max_tregs(self, dt : adt):
        return 0

    def treg(self, reg_idx : int):
        raise NotImplementedError("NEON has no tiles, use SME")

    def zero_treg(self, treg : treg_type, datatype : adt):
        raise NotImplementedError("NEON has no tiles, use SME")

    def store_tile(self, areg : greg_type,
                   ignored_offset : int,
                   treg : treg_type,
                   datatype : adt):
        raise NotImplementedError("NEON has no tiles, use SME")
