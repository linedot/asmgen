from asmgen.asmblocks.noarch import asmgen, reg_tracker
from asmgen.asmblocks.noarch import asm_data_type,asm_index_type
from asmgen.asmblocks.noarch import greg, freg, vreg

from abc import abstractmethod

import sys
if not sys.version_info >= (3, 10):
    from typing_extensions import TypeAlias
else:
    from typing import TypeAlias
from typing import Union

class x86_greg(greg):
    def __init__(self, reg_idx : int):
        self.reg_str = avxbase.greg_names[reg_idx]

    def __str__(self) -> str:
        return self.reg_str

class avx_freg(freg):
    def __init__(self, reg_idx : int):
        self.reg_str = f"xmm{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str

class xmm_vreg(vreg):
    def __init__(self, reg_idx : int):
        self.reg_str = f"xmm{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str

class ymm_vreg(vreg):
    def __init__(self, reg_idx : int):
        self.reg_str = f"ymm{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str

class zmm_vreg(vreg):
    def __init__(self, reg_idx : int):
        self.reg_str = f"zmm{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str

class avxbase(asmgen):

    greg_type : TypeAlias = greg
    freg_type : TypeAlias = freg
    vreg_type : TypeAlias = vreg

    @property
    @abstractmethod
    def req_flags(self) -> list[str]:
        raise NotImplementedError("Not implemented for base class")

    def supportedby_cpuinfo(self, cpuinfo : str):
         req_flags = self.req_flags
         supported = True
         for r in req_flags:
             if -1 == cpuinfo.find(r):
                 supported = False
                 break
         return supported

    dt_suffixes_packed = {
            asm_data_type.DOUBLE : "pd",
            asm_data_type.SINGLE : "ps",
            }

    it_suffixes = {
            asm_index_type.INT64 : "q",
            asm_index_type.INT32 : "d",
            }

    dt_suffixes_single = {
            asm_data_type.DOUBLE : "sd",
            asm_data_type.SINGLE : "ss",
            }

    greg_names = [f'r{i}' for i in [str(j) for j in range(8,16)]+['ax','bx','cx','dx','si','di','bp','sp']]

    def isaquirks(self, rt : reg_tracker, dt : asm_data_type):
        return ""

    def jzero(self, greg : greg_type, label : str) -> str:
        preg = self.prefix_if_raw_reg(greg)
        asmblock  = self.asmwrap(f"test {preg},{preg}")
        asmblock += self.asmwrap(f"jz .{label}%=")
        return asmblock

    def jfzero(self, freg1 : freg_type, freg2 : freg_type,
               greg : greg_type, label : str,
               datatype : asm_data_type) -> str:
        suf = self.dt_suffixes_single[datatype]
        pfreg1 = self.prefix_if_raw_reg(freg1)
        pfreg2 = self.prefix_if_raw_reg(freg2)
        asmblock  = self.zero_freg(freg2, datatype)
        asmblock += self.asmwrap(f"ucomi{suf} {pfreg2},{pfreg1}")
        asmblock += self.asmwrap(f"je .{label}%=")
        return asmblock

    def jvzero(self, vreg1 : vreg_type, freg : freg_type,
               vreg2 : vreg_type, greg : greg_type, label : str,
               datatype : asm_data_type) -> str:
        suf = self.dt_suffixes_packed[datatype]
        pvreg1 = self.prefix_if_raw_reg(vreg1)
        pvreg2 = self.prefix_if_raw_reg(vreg2)
        asmblock  = self.zero_vreg(vreg2, datatype)
        asmblock += self.asmwrap(f"vcmpeq{suf} {pvreg2},{pvreg1},{pvreg2}")
        asmblock += self.asmwrap(f"vptest {pvreg2},{pvreg2}")
        asmblock += self.asmwrap(f"jne .{label}%=")
        return asmblock

    def prefix_if_raw_reg(self,  reg : Union[greg_type,
                                             freg_type,
                                             vreg_type]) -> str:
        # If there is a [ it's probably a parameter
        if '[' in str(reg):
            return str(reg)
        return f"%%{reg}"

    def xmm_to_ymm(self, vreg):
        return vreg.replace("xmm","ymm")

    @property
    def are_fregs_in_vregs(self):
        return True

    def label(self, label : str) -> str:
        asmblock = self.asmwrap(f".{label}%=:")
        return asmblock

    def jump(self, label : str) -> str:
        asmblock = self.asmwrap(f"jmp .{label}%=")
        return asmblock

    def loopbegin(self, reg : greg_type, label : str):
        preg = self.prefix_if_raw_reg(reg)
        asmblock  = self.asmwrap(f".{label}%=:")
        asmblock += self.asmwrap(f"sub $1, {preg}")
        return asmblock

    def loopbegin_nz(self, reg : greg_type, label : str, skiplabel : str):
        preg = self.prefix_if_raw_reg(reg)
        asmblock  = self.asmwrap(f"test {preg},{preg}")
        asmblock += self.asmwrap(f"jz .{skiplabel}%=")
        asmblock += self.asmwrap(f".{label}%=:")
        asmblock += self.asmwrap(f"sub $1, {preg}")
        return asmblock

    def loopend(self, reg, label):
        preg = self.prefix_if_raw_reg(reg)
        asmblock  = self.asmwrap(f"cmp $0x0,{preg}")
        asmblock += self.asmwrap(f"jne .{label}%=")

        return asmblock

    @property
    def is_vla(self):
        return False

    def indexable_elements(self, datatype):
        return self.simd_size/datatype.value

    @property
    def max_gregs(self):
        return 16

    @property
    def c_simd_size_function(self):
        result  = f"size_t get_simd_size() {{ return {self.simd_size}; }}"
        return result


    def fma(self, a, b, c, datatype):
        suf = self.dt_suffixes_packed[datatype]
        pa = self.prefix_if_raw_reg(a)
        pb = self.prefix_if_raw_reg(b)
        pc = self.prefix_if_raw_reg(c)
        return self.asmwrap(f"vfmadd231{suf} {pa},{pb},{pc}")

    def fma_np(self, a, b, c, datatype):
        suf = self.dt_suffixes_packed[datatype]
        pa = self.prefix_if_raw_reg(a)
        pb = self.prefix_if_raw_reg(b)
        pc = self.prefix_if_raw_reg(c)
        return self.asmwrap(f"vfnmadd231{suf} {pa},{pb},{pc}")

    def fmul(self, a, b, c, datatype):
        suf = self.dt_suffixes_packed[datatype]
        pa = self.prefix_if_raw_reg(a)
        pb = self.prefix_if_raw_reg(b)
        pc = self.prefix_if_raw_reg(c)
        return self.asmwrap(f"vmul{suf} {pa},{pb},{pc}")

    def fma_vf(self, a, b, c, datatype):
        raise NotImplementedError("AVX-FMA doesn't have vector x scalar FMA instruction")

    def fma_np_vf(self, a, b, c, datatype):
        raise NotImplementedError("AVX-FMA doesn't have vector x scalar FMA instruction")

    def fmul_vf(self, a, b, c, datatype):
        raise NotImplementedError("AVX-FMA doesn't have vector x scalar FMUL instruction")

    def fma_idx(self, a, b, c, idx, datatype):
        raise NotImplementedError("Indexed FMA not implemented for AVX-FMA generators")

    def fma_np_idx(self, a, b, c, idx, datatype):
        raise NotImplementedError("Indexed FMA not implemented for AVX-FMA generators")

    def mov_greg(self, src, dst):
        psrc = self.prefix_if_raw_reg(src)
        pdst = self.prefix_if_raw_reg(dst)
        return self.asmwrap(f"movq {psrc}, {pdst}")

    def mov_freg(self, src, dst, datatype : asm_data_type):
        suf = self.dt_suffixes_single[datatype]
        return self.asmwrap(f"vmov{suf} {src}, {dst}")

    def mov_greg_to_param(self, reg, param):
        preg = self.prefix_if_raw_reg(reg)
        return self.asmwrap(f"movq {preg},%[{param}]")

    def mov_param_to_greg(self, param, reg):
        preg = self.prefix_if_raw_reg(reg)
        return self.asmwrap(f"movq %[{param}],{preg}")

    def mov_param_to_greg_shift(self, param, dst, offset):
        pdst = self.prefix_if_raw_reg(dst)
        return self.asmwrap(f"leaq (,%[{param}],{1<<offset}),{pdst}")

    def mov_greg_imm(self, reg, imm):
        preg = self.prefix_if_raw_reg(reg)
        return self.asmwrap(f"movq ${imm},{preg}")

    def zero_greg(self, reg):
        return self.mov_greg_imm(reg, 0)

    def add_greg_imm(self, reg, offset):
        preg = self.prefix_if_raw_reg(reg)
        return self.asmwrap(f"addq ${offset},{preg}")

    def mul_greg_imm(self, src, dst, offset):
        pdst = self.prefix_if_raw_reg(dst)
        psrc = self.prefix_if_raw_reg(src)
        return self.asmwrap(f"imulq ${offset},{psrc},{pdst}")

    def add_greg_greg(self, dst, reg1, reg2):
        pdst = self.prefix_if_raw_reg(dst)
        preg1 = self.prefix_if_raw_reg(reg1)
        preg2 = self.prefix_if_raw_reg(reg2)
        # Eh... might be inefficient in some weird
        # cases but I can't be bothered to handle x86 differently
        asmblock  = self.asmwrap(f"movq {preg1},{pdst}")
        asmblock += self.asmwrap(f"addq {preg2},{pdst}")
        return asmblock

    @property
    def has_add_greg_voff(self) -> bool:
        return True

    def add_greg_voff(self, reg, offset, datatype):
        offset = offset*self.simd_size
        return self.add_greg_imm(reg, offset)

    def shift_greg_left(self, reg, offset):
        preg = self.prefix_if_raw_reg(reg)
        return self.asmwrap(f"shlq ${offset},{preg}")

    def shift_greg_right(self, reg, offset):
        preg = self.prefix_if_raw_reg(reg)
        return self.asmwrap(f"shrq ${offset},{preg}")

    def greg(self, i):
        return self.greg_names[i]

    def freg(self, reg_idx : int) -> freg_type:
        return avx_freg(reg_idx)
        #return f"xmm{i}"

    def zero_freg(self,reg,datatype):
        return self.zero_vreg(reg, datatype)

    @property
    def min_prefetch_offset(self):
        return 0

    @property
    def max_prefetch_offset(self):
        return 2**31

    def min_load_immoff(self,datatype):
        return 0

    def max_load_immoff(self,datatype):
        return 2**31

    @property
    def max_fload_immoff(self):
        return 2**31

    @property
    def min_load_voff(self):
        return 0

    @property
    def max_load_voff(self):
        return 2**31/self.simd_size

    def prefetch_l1_boff(self, reg, offset):
        preg = self.prefix_if_raw_reg(reg)
        return self.asmwrap(f"prefetcht0 {offset}({preg})")

    def load_pointer(self, reg, name):
        preg = self.prefix_if_raw_reg(reg)
        return self.asmwrap(f"mov %[{name}],{preg}")


    def load_vector_voff(self, a, voffset, v, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        suf = self.dt_suffixes_packed[datatype]
        pa = self.prefix_if_raw_reg(a)
        pv = self.prefix_if_raw_reg(v)
        address = f"{voffset*self.simd_size}({pa})"
        if 0 == voffset:
            address = f"({pa})"
        return self.asmwrap(f"vmovu{suf} {address},{pv}")

    def load_scalar_immoff(self, a, immoffset, v, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        suf = self.dt_suffixes_single[datatype]
        pa = self.prefix_if_raw_reg(a)
        pv = self.prefix_if_raw_reg(v)
        address = f"{immoffset}({pa})"
        if 0 == immoffset:
            address = f"({pa})"
        return self.asmwrap(f"vmov{suf} {address},{pv}")

    def load_vector(self, a, ignored_offset, v, datatype):
        return self.load_vector_voff(a, 0, v, datatype)

    def load_vector_dist1_inc(self, a, ignored_offset, v, datatype):
        raise NotImplementedError("AVX doesn't have a post-index load, use load_vector_dist1_boff instead")

    def store_vector_voff(self, a, voffset, v, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        suf = self.dt_suffixes_packed[datatype]
        pa = self.prefix_if_raw_reg(a)
        pv = self.prefix_if_raw_reg(v)
        address = f"{voffset*self.simd_size}({pa})"
        if 0 == voffset:
            address = f"({pa})"
        return self.asmwrap(f"vmovu{suf} {pv},{address}")

    def store_vector(self, a, voffset, v, datatype):
        return self.store_vector_voff(a, 0, v, datatype)


    def load_vector_immstride(self, areg : greg_type, byte_stride : int,
                    vreg : vreg_type, datatype : asm_data_type):
        raise NotImplementedError("AVX has no load with immediate stride")

    def load_vector_gregstride(self, areg : greg_type, sreg : greg_type,
                    vreg : vreg_type, datatype : asm_data_type):
        raise NotImplementedError("AVX has no load with scalar register stride")

    def load_vector_gather(self, areg : greg_type, offvreg : vreg_type,
                           vreg : vreg_type, datatype : asm_data_type,
                           indextype : asm_index_type):
        suf = self.dt_suffixes_single[datatype]
        pa = self.prefix_if_raw_reg(areg)
        pv = self.prefix_if_raw_reg(vreg)
        pov = self.prefix_if_raw_reg(offvreg)
        address = f"({pa})"
        isuf = self.it_suffixes[indextype]
        return self.asmwrap(f"vgather{isuf}{suf} {address},{pov},{pv}")

    def store_vector_immstride(self, areg : greg_type, byte_stride : int,
                    vreg : vreg_type, datatype : asm_data_type):
        raise NotImplementedError("AVX has no store with immediate stride")

    def store_vector_gregstride(self, areg : greg_type, sreg : greg_type,
                    vreg : vreg_type, datatype : asm_data_type):
        raise NotImplementedError("AVX has no store with scalar register stride")

    def store_vector_scatter(self, areg : greg_type, offvreg : vreg_type,
                    vreg : vreg_type, datatype : asm_data_type):
        raise NotImplementedError("AVX has no store with vector register stride")

class fma128(avxbase):

    @property
    def req_flags(self):
        return ['fma', 'avx']

    @property
    def max_fregs(self):
        return 16
    
    @property
    def max_vregs(self):
        return 16

    @property
    def simd_size(self):
        return 16

    def zero_vreg(self,reg,datatype):
        preg = self.prefix_if_raw_reg(reg)
        return self.asmwrap(f"vpxor {preg},{preg},{preg}")

    def vreg(self, reg_idx):
        return xmm_vreg(reg_idx)

    def load_vector_dist1(self, a, ignored_offset, v, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        suf = self.dt_suffixes_single[datatype]
        pa = self.prefix_if_raw_reg(a)
        pv = self.xmm_to_ymm(self.prefix_if_raw_reg(v))
        return self.asmwrap(f"vbroadcast{suf} ({pa}),{pv}")

    def load_vector_dist1_boff(self, a, offset, v, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        suf = self.dt_suffixes_single[datatype]
        pa = self.prefix_if_raw_reg(a)
        pv = self.xmm_to_ymm(self.prefix_if_raw_reg(v))
        return self.asmwrap(f"vbroadcast{suf} {offset}({pa}),{pv}")

class fma256(avxbase):

    @property
    def req_flags(self):
        return ['fma', 'avx']

    @property
    def max_fregs(self):
        return 16

    @property
    def max_vregs(self):
        return 16

    @property
    def simd_size(self):
        return 32

    def zero_vreg(self,reg,datatype):
        preg = self.prefix_if_raw_reg(reg)
        return self.asmwrap(f"vpxor {preg},{preg},{preg}")

    def vreg(self, reg_idx : int):
        return ymm_vreg(reg_idx)

    def load_vector_dist1(self, a, ignored_offset, v, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        suf = self.dt_suffixes_single[datatype]
        pa = self.prefix_if_raw_reg(a)
        pv = self.prefix_if_raw_reg(v)
        return self.asmwrap(f"vbroadcast{suf} ({pa}),{pv}")

    def load_vector_dist1_boff(self, a, offset, v, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        suf = self.dt_suffixes_single[datatype]
        pa = self.prefix_if_raw_reg(a)
        pv = self.prefix_if_raw_reg(v)
        return self.asmwrap(f"vbroadcast{suf} {offset}({pa}),{pv}")

class avx512(avxbase):

    greg_type : TypeAlias = greg
    freg_type : TypeAlias = freg
    vreg_type : TypeAlias = vreg

    @property
    def req_flags(self):
        return ['avx512f']

    @property
    def max_fregs(self):
        return 32

    @property
    def max_vregs(self):
        return 32

    @property
    def simd_size(self):
        return 64

    def zero_vreg(self, reg, datatype):
        preg = self.prefix_if_raw_reg(reg)
        return self.asmwrap(f"vpxorq {preg},{preg},{preg}")

    def vreg(self, reg_idx : int):
        return zmm_vreg(reg_idx)

    def load_vector_dist1(self, a, ignored_offset, v, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        suf = self.dt_suffixes_single[datatype]
        pa = self.prefix_if_raw_reg(a)
        pv = self.prefix_if_raw_reg(v)
        return self.asmwrap(f"vbroadcast{suf} ({pa}),{pv}")

    def load_vector_dist1_boff(self, a, offset, v, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        suf = self.dt_suffixes_single[datatype]
        pa = self.prefix_if_raw_reg(a)
        pv = self.prefix_if_raw_reg(v)
        return self.asmwrap(f"vbroadcast{suf} {offset}({pa}),{pv}")

    def store_vector_scatter(self, areg : greg_type, offvreg : vreg_type,
                           vreg : vreg_type, datatype : asm_data_type,
                           indextype : asm_index_type):
        suf = self.dt_suffixes_single[datatype]
        pa = self.prefix_if_raw_reg(areg)
        pv = self.prefix_if_raw_reg(vreg)
        pov = self.prefix_if_raw_reg(offvreg)
        address = f"({pa})"
        isuf = self.it_suffixes[indextype]
        return self.asmwrap(f"vscatter{isuf}{suf} {address},{pov},{pv}")

