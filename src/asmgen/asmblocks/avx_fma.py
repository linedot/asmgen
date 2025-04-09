"""
X86_64/AVX/FMA asm generator and related types
"""

from typing import Union
from abc import abstractmethod

from ..registers import (
    reg_tracker,
    asm_data_type as adt,
    adt_size,
    asm_index_type as ait,
    treg_base, vreg_base, freg_base, greg_base
)

from .noarch import asmgen

from .types.avx_types import x86_greg,avx_freg,xmm_vreg,ymm_vreg,zmm_vreg,prefix_if_raw_reg

from .avx_opd3 import avx_fma,avx_fmul

class avxbase(asmgen):
    """
    Base X86_64/AVX/FMA asmgem implementation
    """


    def get_parameters(self) -> list[str]:
        return []

    def set_parameter(self, name : str, value : Union[str,int]):
        raise ValueError(f"Invalid name {name} or value {value}")

    @abstractmethod
    def get_req_flags(self) -> list[str]:
        """
        Return required flags in cpuinfo for this generator to be supported
        """
        raise NotImplementedError("Not implemented for base class")

    def supportedby_cpuinfo(self, cpuinfo : str):
        req_flags = self.get_req_flags()
        supported = True
        for r in req_flags:
            if -1 == cpuinfo.find(r):
                supported = False
                break
        return supported

    dt_suffixes = {
            adt.DOUBLE : "d",
            adt.SINGLE : "s",
            adt.HALF   : "h",
            }

    it_suffixes = {
            ait.INT64 : "q",
            ait.INT32 : "d",
            }


    def isaquirks(self, *, rt : reg_tracker, dt : adt):
        return ""
    def isaendquirks(self, *, rt : reg_tracker, dt : adt):
        return ""

    def jzero(self, *, reg : greg_base, label : str) -> str:
        preg = prefix_if_raw_reg(reg)
        asmblock  = self.asmwrap(f"test {preg},{preg}")
        asmblock += self.asmwrap(f"jz .{label}%=")
        return asmblock

    def jfzero(self, *, freg1 : freg_base, freg2 : freg_base,
               greg : greg_base, label : str,
               dt : adt) -> str:
        suf = 's'+self.dt_suffixes[dt]
        pfreg1 = prefix_if_raw_reg(freg1)
        pfreg2 = prefix_if_raw_reg(freg2)
        asmblock  = self.zero_freg(freg=freg2, dt=dt)
        asmblock += self.asmwrap(f"ucomi{suf} {pfreg2},{pfreg1}")
        asmblock += self.asmwrap(f"je .{label}%=")
        return asmblock

    def jvzero(self, *, vreg1 : vreg_base, freg : freg_base,
               vreg2 : vreg_base, greg : greg_base, label : str,
               dt : adt) -> str:
        suf = 'p'+self.dt_suffixes[dt]
        pvreg1 = prefix_if_raw_reg(vreg1)
        pvreg2 = prefix_if_raw_reg(vreg2)
        asmblock  = self.zero_vreg(vreg=vreg2, dt=dt)
        asmblock += self.asmwrap(f"vcmpeq{suf} {pvreg2},{pvreg1},{pvreg2}")
        asmblock += self.asmwrap(f"vptest {pvreg2},{pvreg2}")
        asmblock += self.asmwrap(f"jne .{label}%=")
        return asmblock

    def xmm_to_ymm(self, vreg : vreg_base):
        """
        Returns the expanded ymm register from the specified xmm register
        
        :param vreg: XMM register to convert to YMM
        :type vreg: class:`asmgen.asmblocks.types.avx_types.xmm_vreg`
        :return: Corresponding YMM register
        :rtype vreg: class:`asmgen.asmblocks.types.avx_types.ymm_vreg`
        """
        if not isinstance(vreg, xmm_vreg):
            raise ValueError(f"{vreg} is not an XMM register")

        idx = 0
        xmm_str = f"{vreg}"
        if xmm_str.startswith("xmm"):
            idx = int(xmm_str[3:])
        elif xmm_str.startswith("%%xmm"):
            idx = int(xmm_str[5:])
        else:
            raise ValueError("Not an XMM register: {vreg}")
        return ymm_vreg(idx)

    @property
    def are_fregs_in_vregs(self):
        return True

    def label(self, *, label : str) -> str:
        asmblock = self.asmwrap(f".{label}%=:")
        return asmblock

    def jump(self, *, label : str) -> str:
        asmblock = self.asmwrap(f"jmp .{label}%=")
        return asmblock

    def loopbegin(self, *, reg : greg_base, label : str):
        preg = prefix_if_raw_reg(reg)
        asmblock  = self.asmwrap(f".{label}%=:")
        asmblock += self.asmwrap(f"sub $1, {preg}")
        return asmblock

    def loopbegin_nz(self, *, reg : greg_base, label : str, labelskip : str):
        preg = prefix_if_raw_reg(reg)
        asmblock  = self.asmwrap(f"test {preg},{preg}")
        asmblock += self.asmwrap(f"jz .{labelskip}%=")
        asmblock += self.asmwrap(f".{label}%=:")
        asmblock += self.asmwrap(f"sub $1, {preg}")
        return asmblock

    def loopend(self, *, reg : greg_base, label : str):
        preg = prefix_if_raw_reg(reg)
        asmblock  = self.asmwrap(f"cmp $0x0,{preg}")
        asmblock += self.asmwrap(f"jne .{label}%=")

        return asmblock

    @property
    def is_vla(self):
        return False

    def indexable_elements(self, dt : adt):
        return self.simd_size//adt_size(dt)

    @property
    def max_gregs(self):
        return 16

    @property
    def c_simd_size_function(self):
        result  = f"size_t get_simd_size() {{ return {self.simd_size}; }}"
        return result

    def simd_size_to_greg(self, *, reg: greg_base, dt: adt) -> str:
        return self.mov_greg_imm(reg=reg, imm=self.simd_size//adt_size(dt))

    def mov_greg(self, *, src : greg_base, dst : greg_base):
        psrc = prefix_if_raw_reg(src)
        pdst = prefix_if_raw_reg(dst)
        return self.asmwrap(f"movq {psrc}, {pdst}")

    def mov_freg(self, *, src : freg_base, dst : freg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        return self.asmwrap(f"vmov{suf} {src}, {dst}")

    def mov_greg_to_param(self, *, src : greg_base, param : str):
        preg = prefix_if_raw_reg(src)
        return self.asmwrap(f"movq {preg},%[{param}]")

    def mov_param_to_greg(self, *, param : str, dst : greg_base):
        preg = prefix_if_raw_reg(dst)
        return self.asmwrap(f"movq %[{param}],{preg}")

    def mov_param_to_greg_shift(self, *, param : str, dst : greg_base, bit_count : int):
        pdst = prefix_if_raw_reg(dst)
        return self.asmwrap(f"leaq (,%[{param}],{1<<bit_count}),{pdst}")

    def mov_greg_imm(self, *, reg : greg_base, imm : int):
        preg = prefix_if_raw_reg(reg)
        return self.asmwrap(f"movq ${imm},{preg}")

    def zero_greg(self, *, greg : greg_base):
        return self.mov_greg_imm(reg=greg, imm=0)

    def add_greg_imm(self, *, reg : greg_base, imm : int):
        preg = prefix_if_raw_reg(reg)
        return self.asmwrap(f"addq ${imm},{preg}")

    def mul_greg_imm(self, *, src : greg_base, dst : greg_base, factor : int) -> str:
        pdst = prefix_if_raw_reg(dst)
        psrc = prefix_if_raw_reg(src)
        return self.asmwrap(f"imulq ${factor},{psrc},{pdst}")

    def add_greg_greg(self, *, dst : greg_base, reg1 : greg_base, reg2 : greg_base) -> str:
        pdst = prefix_if_raw_reg(dst)
        preg1 = prefix_if_raw_reg(reg1)
        preg2 = prefix_if_raw_reg(reg2)
        # Eh... might be inefficient in some weird
        # cases but I can't be bothered to handle x86 differently
        asmblock  = self.asmwrap(f"movq {preg1},{pdst}")
        asmblock += self.asmwrap(f"addq {preg2},{pdst}")
        return asmblock

    def sub_greg_greg(self, *, dst : greg_base, reg1 : greg_base, reg2 : greg_base) -> str:
        pdst = prefix_if_raw_reg(dst)
        preg1 = prefix_if_raw_reg(reg1)
        preg2 = prefix_if_raw_reg(reg2)
        asmblock  = self.asmwrap(f"movq {preg1},{pdst}")
        asmblock += self.asmwrap(f"subq {preg2},{pdst}")
        return asmblock

    def add_greg_voff(self, *, reg : greg_base, offset : int, dt : adt) -> str:
        offset = offset*self.simd_size
        return self.add_greg_imm(reg=reg, imm=offset)

    def shift_greg_left(self, *, reg : greg_base, bit_count : int) -> str:
        preg = prefix_if_raw_reg(reg)
        return self.asmwrap(f"shlq ${bit_count},{preg}")

    def shift_greg_right(self, *, reg : greg_base, bit_count : int) -> str:
        preg = prefix_if_raw_reg(reg)
        return self.asmwrap(f"shrq ${bit_count},{preg}")

    def greg(self, reg_idx : int) -> greg_base:
        return x86_greg(reg_idx=reg_idx)

    def freg(self, reg_idx : int, dt : adt) -> freg_base:
        _ = dt # explicitly unused
        return avx_freg(reg_idx=reg_idx)

    def zero_freg(self, *, freg : freg_base, dt : adt) -> str:
        preg = prefix_if_raw_reg(freg)
        return self.asmwrap(f"vpxor {preg},{preg},{preg}")

    @property
    def min_prefetch_offset(self):
        return 0

    @property
    def max_prefetch_offset(self):
        return 2**31

    def min_load_immoff(self, dt : adt):
        return 0

    def max_load_immoff(self, dt : adt):
        return 2**31

    def min_fload_immoff(self, dt : adt):
        return 0

    def max_fload_immoff(self, dt : adt):
        return 2**31

    @property
    def min_load_voff(self):
        return 0

    @property
    def max_load_voff(self) -> int:
        return 2**31//self.simd_size


    @property
    def max_add_voff(self) -> int:
        return 2**31//self.simd_size

    def prefetch_l1_boff(self, *, areg : greg_base, offset : int):
        preg = prefix_if_raw_reg(areg)
        return self.asmwrap(f"prefetcht0 {offset}({preg})")

    def load_pointer(self, *, areg : greg_base, name : str):
        preg = prefix_if_raw_reg(areg)
        return self.asmwrap(f"mov %[{name}],{preg}")


    def load_vector_voff(self, *, areg : greg_base, voffset : int, vreg : vreg_base, dt : adt):
        suf = 'p'+self.dt_suffixes[dt]
        pa = prefix_if_raw_reg(areg)
        pv = prefix_if_raw_reg(vreg)
        address = f"{voffset*self.simd_size}({pa})"
        if 0 == voffset:
            address = f"({pa})"
        return self.asmwrap(f"vmovu{suf} {address},{pv}")

    def load_scalar_immoff(self, *, areg : greg_base, offset : int, freg : freg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        pa = prefix_if_raw_reg(areg)
        pf = prefix_if_raw_reg(freg)
        address = f"{offset}({pa})"
        if 0 == offset:
            address = f"({pa})"
        return self.asmwrap(f"vmov{suf} {address},{pf}")

    def load_vector(self, *, areg : greg_base, vreg : vreg_base, dt : adt):
        return self.load_vector_voff(areg=areg, voffset=0, vreg=vreg, dt=dt)

    def load_vector_dist1_inc(self, *, areg : greg_base, offset : int, vreg : vreg_base, dt : adt):
        raise NotImplementedError(
                "AVX doesn't have a post-index load, use load_vector_dist1_boff instead")

    def store_vector_voff(self, *, areg : greg_base, voffset : int, vreg : vreg_base, dt : adt):
        suf = 'p'+self.dt_suffixes[dt]
        pa = prefix_if_raw_reg(areg)
        pv = prefix_if_raw_reg(vreg)
        address = f"{voffset*self.simd_size}({pa})"
        if 0 == voffset:
            address = f"({pa})"
        return self.asmwrap(f"vmovu{suf} {pv},{address}")

    def store_vector(self, *, areg : greg_base, vreg : vreg_base, dt : adt):
        return self.store_vector_voff(areg=areg, voffset=0, vreg=vreg, dt=dt)


    def load_vector_immstride(self, *, areg : greg_base, byte_stride : int,
                    vreg : vreg_base, dt : adt):
        raise NotImplementedError("AVX has no load with immediate stride")

    def load_vector_gregstride(self, *, areg : greg_base, sreg : greg_base,
                    vreg : vreg_base, dt : adt):
        raise NotImplementedError("AVX has no load with scalar register stride")

    def load_vector_gather(self, *, areg : greg_base, offvreg : vreg_base,
                           vreg : vreg_base, dt : adt,
                           it : ait):
        suf = 's'+self.dt_suffixes[dt]
        pa = prefix_if_raw_reg(areg)
        pv = prefix_if_raw_reg(vreg)
        pov = prefix_if_raw_reg(offvreg)
        address = f"({pa})"
        isuf = self.it_suffixes[it]
        return self.asmwrap(f"vgather{isuf}{suf} {address},{pov},{pv}")

    def store_vector_immstride(self, *, areg : greg_base, byte_stride : int,
                    vreg : vreg_base, dt : adt):
        raise NotImplementedError("AVX has no store with immediate stride")

    def store_vector_gregstride(self, *, areg : greg_base, sreg : greg_base,
                    vreg : vreg_base, dt : adt):
        raise NotImplementedError("AVX has no store with scalar register stride")

    def store_vector_scatter(self, *, areg : greg_base, offvreg : vreg_base,
                             vreg : vreg_base, dt : adt, it : ait):
        raise NotImplementedError("AVX has no store with vector register stride")

    # Unsupported functionality:
    def max_tregs(self, dt : adt):
        return 0

    def treg(self, reg_idx : int):
        raise NotImplementedError("SVE has no tiles, use SME")

    def zero_treg(self, *, treg : treg_base, dt : adt):
        raise NotImplementedError("SVE has no tiles, use SME")

    def store_tile(self, *, areg : greg_base,
                   treg : treg_base,
                   dt : adt):
        raise NotImplementedError("SVE has no tiles, use SME")

class fma128(avxbase):
    """
    X86_64/AVX/FMA 128 bit asmgem implementation
    """

    def __init__(self):
        super().__init__()
        self.fma = avx_fma(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     has_fp16=False
                     )
        self.fmul = avx_fmul(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     has_fp16=False
                     )

    def get_req_flags(self):
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

    def zero_vreg(self, *, vreg : vreg_base, dt : adt):
        preg = prefix_if_raw_reg(vreg)
        return self.asmwrap(f"vpxor {preg},{preg},{preg}")

    def vreg(self, reg_idx):
        return xmm_vreg(reg_idx)

    def load_vector_dist1(self, *, areg : greg_base,
                          vreg : vreg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        pa = prefix_if_raw_reg(areg)
        pv = prefix_if_raw_reg(self.xmm_to_ymm(vreg))
        return self.asmwrap(f"vbroadcast{suf} ({pa}),{pv}")

    def load_vector_dist1_boff(self, *, areg : greg_base, offset : int,
                               vreg : vreg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        pa = prefix_if_raw_reg(areg)
        pv = prefix_if_raw_reg(self.xmm_to_ymm(vreg))
        return self.asmwrap(f"vbroadcast{suf} {offset}({pa}),{pv}")

class fma256(avxbase):
    """
    X86_64/AVX/FMA 256 bit asmgem implementation
    """

    def __init__(self):
        super().__init__()
        self.fma = avx_fma(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     has_fp16=False
                     )
        self.fmul = avx_fmul(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     has_fp16=False
                     )

    def get_req_flags(self):
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

    def zero_vreg(self, *, vreg : vreg_base, dt : adt):
        preg = prefix_if_raw_reg(vreg)
        return self.asmwrap(f"vpxor {preg},{preg},{preg}")

    def vreg(self, reg_idx : int):
        return ymm_vreg(reg_idx)

    def load_vector_dist1(self, *, areg : greg_base,
                          vreg : vreg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        pa = prefix_if_raw_reg(areg)
        pv = prefix_if_raw_reg(vreg)
        return self.asmwrap(f"vbroadcast{suf} ({pa}),{pv}")

    def load_vector_dist1_boff(self, *, areg : greg_base, offset : int,
                               vreg : vreg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        pa = prefix_if_raw_reg(areg)
        pv = prefix_if_raw_reg(vreg)
        return self.asmwrap(f"vbroadcast{suf} {offset}({pa}),{pv}")

class avx512(avxbase):
    """
    X86_64/AVX512 asmgem implementation
    """

    def __init__(self):
        super().__init__()
        self.fma = avx_fma(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     has_fp16=True
                     )
        self.fmul = avx_fmul(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     has_fp16=True
                     )

    def get_req_flags(self) -> list[str]:
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

    def zero_vreg(self, *, vreg : vreg_base, dt : adt):
        preg = prefix_if_raw_reg(vreg)
        return self.asmwrap(f"vpxorq {preg},{preg},{preg}")

    def vreg(self, reg_idx : int):
        return zmm_vreg(reg_idx)

    def load_vector_dist1(self, *, areg : greg_base,
                          vreg : vreg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        pa = prefix_if_raw_reg(areg)
        pv = prefix_if_raw_reg(vreg)
        return self.asmwrap(f"vbroadcast{suf} ({pa}),{pv}")

    def load_vector_dist1_boff(self, *, areg : greg_base, offset : int,
                               vreg : vreg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        pa = prefix_if_raw_reg(areg)
        pv = prefix_if_raw_reg(vreg)
        return self.asmwrap(f"vbroadcast{suf} {offset}({pa}),{pv}")

    def store_vector_scatter(self, *, areg : greg_base, offvreg : vreg_base,
                           vreg : vreg_base, dt : adt,
                           it : ait):
        suf = 's'+self.dt_suffixes[dt]
        pa = prefix_if_raw_reg(areg)
        pv = prefix_if_raw_reg(vreg)
        pov = prefix_if_raw_reg(offvreg)
        address = f"({pa})"
        isuf = self.it_suffixes[it]
        return self.asmwrap(f"vscatter{isuf}{suf} {address},{pov},{pv}")
