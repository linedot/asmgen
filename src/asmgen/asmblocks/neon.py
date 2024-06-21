from asmgen.asmblocks.noarch import asm_data_type,asm_index_type
from asmgen.asmblocks.noarch import vreg,freg,greg
from asmgen.asmblocks.aarch64 import aarch64

import sys

if not sys.version_info >= (3, 10):
    from typing_extensions import TypeAlias
else:
    from typing import TypeAlias

class neon_vreg(vreg):
    def __init__(self, reg_idx : int):
        self.reg_str = f"v{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str

class neon(aarch64):

    greg_type : TypeAlias = greg
    freg_type : TypeAlias = freg
    vreg_type : TypeAlias = vreg

    def get_req_flags(self):
        return ['asimd']

    def supportedby_cpuinfo(self, cpuinfo) -> bool:
         req_flags = self.get_req_flags()
         supported = True
         for r in req_flags:
             if -1 == cpuinfo.find(r):
                 supported = False
                 break
         return supported

    dt_suffixes = {
            asm_data_type.DOUBLE : "2d",
            asm_data_type.SINGLE : "4s",
            }

    dt_prefixes = {
            asm_data_type.DOUBLE : "d",
            asm_data_type.SINGLE : "s",
            }

    dt_greg_pfx = {
            asm_data_type.DOUBLE : "x",
            asm_data_type.SINGLE : "w",
            }

    def jvzero(self, vreg1 : vreg_type, freg : freg_type,
               vreg2 : vreg_type,
               greg : greg_type, label : str,
               datatype : asm_data_type):
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

    def indexable_elements(self, datatype):
        return self.simd_size/datatype.value

    @property
    def max_vregs(self):
        return 32

    @property
    def simd_size(self):
        return 16

    @property
    def c_simd_size_function(self):
        return f"size_t get_simd_size() {{ return {self.simd_size}; }}"

    def fmul(self, a, b, c, datatype):
        suf = self.dt_suffixes[datatype]
        return self.asmwrap(f"fmul {c}.{suf},{a}.{suf},{b}.{suf}")

    def fma(self, a, b, c, datatype):
        suf = self.dt_suffixes[datatype]
        return self.asmwrap(f"fmla {c}.{suf},{a}.{suf},{b}.{suf}")

    def fma_idx(self, a, b, c, idx, datatype):
        suf = self.dt_suffixes[datatype]
        sufidx = self.dt_prefixes[datatype]
        return self.asmwrap(f"fmla {c}.{suf},{a}.{suf},{b}.{sufidx}[{idx}]")

    def add_greg_voff(self, reg, offset, datatype):
        byte_offset = self.simd_size*offset
        return self.asmwrap(f"add {reg},{reg},#{byte_offset}")
        
    def zero_vreg(self, reg, datatype):
        suf = self.dt_suffixes[datatype]
        zeroreg = f"{self.dt_greg_pfx[datatype]}zr" 
        return self.asmwrap(f"dup {reg}.{suf},{zeroreg}")

    def vreg(self, reg_idx : int) -> neon_vreg:
        return neon_vreg(reg_idx)

    def qreg(self, i):
        return f"q{i}"

    def min_load_immoff(self,datatype):
        return 0

    def max_load_immoff(self,datatype):
        return 4095*datatype.value*2

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

    def load_vector(self, a, ignored_offset, v, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        qv = self.vreg_to_qreg(v)
        return self.asmwrap(f"ldr {qv}, [{a}]")

    def load_vector_voff(self, a, voffset, v, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        qv = self.vreg_to_qreg(v)
        return self.asmwrap(f"ldr {qv}, [{a}, #{voffset*self.simd_size}]")

    def load_vector_dist1(self, a, ignored_offset, v, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        suf = self.dt_suffixes[datatype]
        return self.asmwrap(f"ld1r {{{v}.{suf}}}, [{a}]")

    def load_vector_dist1_boff(self, a, offset, v, datatype):
        raise NotImplementedError("load_vector_dist1_boff doesn't make sense with NEON, use load_vector_dist1_inc or load_vector_voff + fma_idx instead")
    
    def load_vector_dist1_inc(self, a, ignored_offset, v, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        suf = self.dt_suffixes[datatype]
        return self.asmwrap(f"ld1r {{{v}.{suf}}}, [{a}], #{datatype.value}")

    def store_vector_voff(self, a, voffset, v, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        qv = self.vreg_to_qreg(v)
        return self.asmwrap(f"str {qv}, [{a}, #{voffset*self.simd_size}]")

    def store_vector(self, a, voffset, v, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        qv = self.vreg_to_qreg(v)
        return self.asmwrap(f"str {qv}, [{a}]")

    def load_vector_immstride(self, areg : greg_type, byte_stride : int,
                    vreg : vreg_type, datatype : asm_data_type):
        raise NotImplementedError("NEON has no load with immediate stride")

    def load_vector_gregstride(self, areg : greg_type, sreg : greg_type,
                    vreg : vreg_type, datatype : asm_data_type):
        raise NotImplementedError("NEON has no load with scalar register stride")

    def load_vector_gather(self, areg : greg_type, offvreg : vreg_type,
                           vreg : vreg_type, datatype : asm_data_type,
                           indextype : asm_index_type):
        raise NotImplementedError("NEON has no load with vector register stride")

    def store_vector_immstride(self, areg : greg_type, byte_stride : int,
                    vreg : vreg_type, datatype : asm_data_type):
        raise NotImplementedError("NEON has no store with immediate stride")

    def store_vector_gregstride(self, areg : greg_type, sreg : greg_type,
                    vreg : vreg_type, datatype : asm_data_type):
        raise NotImplementedError("NEON has no store with scalar register stride")

    def store_vector_scatter(self, areg : greg_type, offvreg : vreg_type,
                             vreg : vreg_type, datatype : asm_data_type,
                             indextype : asm_index_type):
        raise NotImplementedError("NEON has no store with vector register stride")
