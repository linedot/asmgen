# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
X86_64/AVX/FMA asm generator and related types
"""

from copy import deepcopy
from typing import Union
from abc import abstractmethod

from ..asmdata import asm_data
from ..registers import (
    reg_tracker,
    asm_data_type as adt,
    adt_size,
    asm_index_type as ait,
    treg_base, vreg_base, freg_base, greg_base
)

from .noarch import asmgen,comparison
from ..callconv.callconv import callconv

from .types.avx_types import x86_greg,avx_freg,xmm_vreg,ymm_vreg,zmm_vreg,reg_prefixer

from .avx_opd3 import avx_fma,avx_fmul

class avxbase(asmgen):
    """
    Base X86_64/AVX/FMA asmgem implementation
    """

    greg_names = [f'r{i}' for i in \
            [str(j) for j in range(8,16)]+\
            ['ax','bx','cx','dx','si','di','bp','sp']]

    cb_insts = {
            'nz' : 'jnz',
            'ez' : 'jz',
            'ne' : 'jne',
            'eq' : 'je',
            'le' : 'jle',
            'ge' : 'jge',
            'lt' : 'jl',
            'gt' : 'jg',
            }

    def __init__(self):
        super().__init__()
        self.callconvs = {
            "systemv" : callconv(
                param_regs={ 'greg' : [13,12,11,10, 0,1], # RDI, RSI, RDX, RCX, R8, R9
                             'freg' : list(range(0, 8))},
                caller_save_lists={ 'greg' : [13,12,11,10, 0,1],  # func args/return vals
                                    'freg' : list(range(0, 8))
                                   },
                callee_save_lists={ 'greg' : list(range(4,8)) +# R12-R15
                                        [9,   # BX
                                         14,  # Frame Pointer
                                         15], # SP
                                    'freg' : list(range(8,16))},
                spreg=15)
            }
        self.default_callconv = "systemv"

        self.rpref = reg_prefixer(
                output_inline_getter=lambda : self.output_inline)

    def create_callconv(self, name : str = "default"):

        if "default" == name:
            name = self.default_callconv

        return deepcopy(self.callconvs[name])

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
            adt.SINT8  : "b",
            adt.SINT16 : "w",
            adt.SINT32 : "d",
            adt.SINT64 : "q",
            }
    size_suffixes = {
            1 : 'b',
            2 : 'w',
            4 : 'd',
            8 : 'q',
            }

    it_suffixes = {
            ait.INT64 : "q",
            ait.INT32 : "d",
            }

    def iota_label(self, size : int, count : int):
        return f"iota_{size*8}x{count}"

    def ensure_indices(self, dt : adt, count : int):
        dt_size = adt_size(dt)
        key = self.iota_label(dt_size, count)
        if key in self.asmdata:
            return

        #TODO: Might want to rework this, maybe ait param and
        #      then introduce ait_to_adt or something?
        index_dt : adt = None
        if dt_size == 1:
            index_dt = adt.SINT8
        elif dt_size == 2:
            index_dt = adt.SINT16
        elif dt_size == 4:
            index_dt = adt.SINT32
        elif dt_size == 8:
            index_dt = adt.SINT64
        else:
            raise ValueError(f"Can't determine index type for {dt}")
            

        self.asmdata[key] = [asm_data(index_dt, i) for i in range(count)]


    def isaquirks(self, *, rt : reg_tracker, dt : adt):
        return ""
    def isaendquirks(self, *, rt : reg_tracker, dt : adt):
        return ""

    def jzero(self, *, reg : greg_base, label : str) -> str:
        preg = self.rpref(reg)
        asmblock  = self.asmwrap(f"test {preg},{preg}")
        asmblock += self.asmwrap(f"jz {self.labelstr(label)}")
        return asmblock

    def jfzero(self, *, freg1 : freg_base, freg2 : freg_base,
               greg : greg_base, label : str,
               dt : adt) -> str:
        suf = 's'+self.dt_suffixes[dt]
        pfreg1 = self.rpref(freg1)
        pfreg2 = self.rpref(freg2)
        asmblock  = self.zero_freg(freg=freg2, dt=dt)
        asmblock += self.asmwrap(f"ucomi{suf} {pfreg2},{pfreg1}")
        asmblock += self.asmwrap(f"je {self.labelstr(label)}")
        return asmblock

    def jvzero(self, *, vreg1 : vreg_base, freg : freg_base,
               vreg2 : vreg_base, greg : greg_base, label : str,
               dt : adt) -> str:
        suf = 'p'+self.dt_suffixes[dt]
        pvreg1 = self.rpref(vreg1)
        pvreg2 = self.rpref(vreg2)
        asmblock  = self.zero_vreg(vreg=vreg2, dt=dt)
        asmblock += self.asmwrap(f"vcmpeq{suf} {pvreg2},{pvreg1},{pvreg2}")
        asmblock += self.asmwrap(f"vptest {pvreg2},{pvreg2}")
        asmblock += self.asmwrap(f"jne {self.labelstr(label)}")
        return asmblock

    def cb(self, *, reg1: greg_base, reg2: greg_base,
           cmp: comparison, label: str) -> str:
        inst = self.cb_insts[cmp.name]
        if reg2 is None:
            pr1 = self.rpref(reg1)
            return self.asmwrap(f"test {pr1},{pr1}")+\
                   self.asmwrap(f"{inst} {self.labelstr(label)}")
        else:
            pr1 = self.rpref(reg1)
            pr2 = self.rpref(reg2)
            return self.asmwrap(f"cmp {pr2},{pr1}")+\
                   self.asmwrap(f"{inst} {self.labelstr(label)}")

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
        elif xmm_str.startswith("%xmm"):
            idx = int(xmm_str[4:])
        elif xmm_str.startswith("%%xmm"):
            idx = int(xmm_str[5:])
        else:
            raise ValueError("Not an XMM register: {vreg}")
        return ymm_vreg(idx)

    @property
    def are_fregs_in_vregs(self):
        return True

    def label(self, *, label : str) -> str:
        asmblock = self.asmwrap(f"{self.labelstr(label)}:")
        return asmblock

    def jump(self, *, label : str) -> str:
        asmblock = self.asmwrap(f"jmp {self.labelstr(label)}")
        return asmblock

    def loopbegin(self, *, reg : greg_base, label : str):
        preg = self.rpref(reg)
        asmblock  = self.asmwrap(f"{self.labelstr(label)}:")
        asmblock += self.asmwrap(f"sub $1, {preg}")
        return asmblock

    def loopbegin_nz(self, *, reg : greg_base, label : str, labelskip : str):
        preg = self.rpref(reg)
        asmblock  = self.asmwrap(f"test {preg},{preg}")
        asmblock += self.asmwrap(f"jz {self.labelstr(labelskip)}")
        asmblock += self.asmwrap(f"{self.labelstr(label)}:")
        asmblock += self.asmwrap(f"sub $1, {preg}")
        return asmblock

    def loopend(self, *, reg : greg_base, label : str):
        preg = self.rpref(reg)
        asmblock  = self.asmwrap(f"cmp $0x0,{preg}")
        asmblock += self.asmwrap(f"jne {self.labelstr(label)}")

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
        result  = f"inline size_t get_simd_size() {{ return {self.simd_size}; }}"
        return result

    def simd_size_to_greg(self, *, reg: greg_base, dt: adt) -> str:
        return self.mov_greg_imm(reg=reg, imm=self.simd_size//adt_size(dt))

    def mov_greg(self, *, src : greg_base, dst : greg_base):
        psrc = self.rpref(src)
        pdst = self.rpref(dst)
        return self.asmwrap(f"movq {psrc}, {pdst}")

    def mov_freg(self, *, src : freg_base, dst : freg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        return self.asmwrap(f"vmov{suf} {src}, {dst}")


    def load_greg(self, *, areg : greg_base, offset : int, dst : greg_base) -> str:
        pareg = self.rpref(areg)
        pdst = self.rpref(dst)
        address = f"{offset}({pareg})"
        if 0 == offset:
            address = f"({pareg})"
        return self.asmwrap(f"movq {address},{pdst}")

    def store_greg(self, *, areg : greg_base, offset : int, src : greg_base) -> str:
        pareg = self.rpref(areg)
        psrc = self.rpref(src)
        address = f"{offset}({pareg})"
        if 0 == offset:
            address = f"({pareg})"
        return self.asmwrap(f"movq {psrc},{address}")

    def mov_greg_to_param(self, *, src : greg_base, param : str):
        preg = self.rpref(src)
        return self.asmwrap(f"movq {preg},%[{param}]")

    def mov_param_to_greg(self, *, param : str, dst : greg_base):
        preg = self.rpref(dst)
        return self.asmwrap(f"movq %[{param}],{preg}")

    def mov_param_to_greg_shift(self, *, param : str, dst : greg_base, bit_count : int):
        pdst = self.rpref(dst)
        return self.asmwrap(f"leaq (,%[{param}],{1<<bit_count}),{pdst}")

    def mov_greg_imm(self, *, reg : greg_base, imm : int):
        preg = self.rpref(reg)
        return self.asmwrap(f"movq ${imm},{preg}")

    def zero_greg(self, *, greg : greg_base):
        return self.mov_greg_imm(reg=greg, imm=0)

    def add_greg_imm(self, *, reg : greg_base, imm : int):
        preg = self.rpref(reg)
        return self.asmwrap(f"addq ${imm},{preg}")

    def mul_greg_imm(self, *, src : greg_base, dst : greg_base, factor : int) -> str:
        pdst = self.rpref(dst)
        psrc = self.rpref(src)
        return self.asmwrap(f"imulq ${factor},{psrc},{pdst}")

    def mul_greg_greg(self, *, dst : greg_base, reg1 : greg_base, reg2 : greg_base) -> str:
        pdst = self.rpref(dst)
        preg1 = self.rpref(reg1)
        preg2 = self.rpref(reg2)
        asmblock = self.asmwrap(f"movq {preg1},{pdst}")
        asmblock += self.asmwrap(f"imulq {preg2},{pdst}")
        return asmblock

    def add_greg_greg(self, *, dst : greg_base, reg1 : greg_base, reg2 : greg_base) -> str:
        pdst = self.rpref(dst)
        preg1 = self.rpref(reg1)
        preg2 = self.rpref(reg2)
        # Eh... might be inefficient in some weird
        # cases but I can't be bothered to handle x86 differently
        asmblock  = self.asmwrap(f"movq {preg1},{pdst}")
        asmblock += self.asmwrap(f"addq {preg2},{pdst}")
        return asmblock

    def sub_greg_greg(self, *, dst : greg_base, reg1 : greg_base, reg2 : greg_base) -> str:
        pdst = self.rpref(dst)
        preg1 = self.rpref(reg1)
        preg2 = self.rpref(reg2)
        asmblock  = self.asmwrap(f"movq {preg1},{pdst}")
        asmblock += self.asmwrap(f"subq {preg2},{pdst}")
        return asmblock

    def add_greg_voff(self, *, reg : greg_base, offset : int, dt : adt) -> str:
        offset = offset*self.simd_size
        return self.add_greg_imm(reg=reg, imm=offset)

    def shift_greg_left(self, *, reg : greg_base, bit_count : int) -> str:
        preg = self.rpref(reg)
        return self.asmwrap(f"shlq ${bit_count},{preg}")

    def shift_greg_right(self, *, reg : greg_base, bit_count : int) -> str:
        preg = self.rpref(reg)
        return self.asmwrap(f"shrq ${bit_count},{preg}")

    def greg(self, reg_idx : int) -> greg_base:
        return x86_greg(reg_idx=reg_idx)

    def freg(self, reg_idx : int, dt : adt) -> freg_base:
        _ = dt # explicitly unused
        return avx_freg(reg_idx=reg_idx)

    def zero_freg(self, *, freg : freg_base, dt : adt) -> str:
        preg = self.rpref(freg)
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

    def min_bcast_immoff(self, dt : adt) -> int:
        return self.min_load_immoff(dt)

    def max_bcast_immoff(self, dt : adt) -> int:
        return self.max_load_immoff(dt)

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

    def greg_to_voffs(self, *, streg : greg_base, vreg : vreg_base, dt : adt) -> str:

        dt_size = adt_size(dt)
        index_count = self.simd_size//adt_size(dt)
        self.ensure_indices(dt, index_count)
        label = self.iota_label(dt_size, index_count)
        label = self.labelstr(label)

        pst = self.rpref(streg,size=dt_size)
        pv = self.rpref(vreg)

        suf = self.size_suffixes[dt_size]
        result  = self.asmwrap(f"vpbroadcast{suf} {pst},{pv}")
        #TODO: properly handle RIP
        rip = '%rip'
        if self.output_inline:
            rip = '%%rip'
        result += self.asmwrap(f"vpmull{suf} {label}({rip}),{pv},{pv}")

        return result

    def prefetch_l1_immoff(self, *, areg : greg_base, offset : int):
        preg = self.rpref(areg)
        return self.asmwrap(f"prefetcht0 {offset}({preg})")

    def load_pointer(self, *, areg : greg_base, name : str):
        preg = self.rpref(areg)
        return self.asmwrap(f"mov %[{name}],{preg}")


    def load_vector_voff(self, *, areg : greg_base, voffset : int, vreg : vreg_base, dt : adt):
        suf = 'p'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pv = self.rpref(vreg)
        address = f"{voffset*self.simd_size}({pa})"
        if 0 == voffset:
            address = f"({pa})"
        return self.asmwrap(f"vmovu{suf} {address},{pv}")

    def load_vector_immoff(self, *, areg : greg_base, offset : int, vreg : vreg_base, dt : adt):
        suf = 'p'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pv = self.rpref(vreg)
        address = f"{offset}({pa})"
        if 0 == offset:
            address = f"({pa})"
        return self.asmwrap(f"vmovu{suf} {address},{pv}")

    def load_scalar_immoff(self, *, areg : greg_base, offset : int, freg : freg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pf = self.rpref(freg)
        address = f"{offset}({pa})"
        if 0 == offset:
            address = f"({pa})"
        return self.asmwrap(f"vmov{suf} {address},{pf}")

    def store_scalar_immoff(self, *, areg : greg_base, offset : int, freg : freg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pf = self.rpref(freg)
        address = f"{offset}({pa})"
        if 0 == offset:
            address = f"({pa})"
        return self.asmwrap(f"vmov{suf} {pf},{address}")

    def load_freg(self, *, areg : greg_base, offset : int, dst: freg_base, dt : adt):
        return self.load_scalar_immoff(areg=areg, offset=offset, freg=dst, dt=dt)

    def store_freg(self, *, areg : greg_base, offset : int, src: freg_base, dt : adt):
        return self.store_scalar_immoff(areg=areg, offset=offset, freg=src, dt=dt)

    def load_vector(self, *, areg : greg_base, vreg : vreg_base, dt : adt):
        return self.load_vector_voff(areg=areg, voffset=0, vreg=vreg, dt=dt)

    def load_vector_bcast1_inc(self, *, areg : greg_base, offset : int, vreg : vreg_base, dt : adt):
        raise NotImplementedError(
                "AVX doesn't have a post-index load, use load_vector_bcast1_immoff instead")

    def store_vector_immoff(self, *, areg : greg_base, offset : int, vreg : vreg_base, dt : adt):
        suf = 'p'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pv = self.rpref(vreg)
        address = f"{offset}({pa})"
        if 0 == offset:
            address = f"({pa})"
        return self.asmwrap(f"vmovu{suf} {pv},{address}")

    def store_vector_voff(self, *, areg : greg_base, voffset : int, vreg : vreg_base, dt : adt):
        suf = 'p'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pv = self.rpref(vreg)
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
        suf = 'p'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pv = self.rpref(vreg)
        pov = self.rpref(offvreg)
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


    avx_no_tile_message = "Not tile support in AVX (will be supported with AMX)"

    def treg(self, reg_idx : int, dt : adt):
        raise NotImplementedError(avx_no_tile_message)

    def zero_treg(self, *, treg : treg_base, dt : adt):
        raise NotImplementedError(avx_no_tile_message)

    def load_tile(self, *, areg : greg_base,
                   treg : treg_base,
                   dt : adt):
        raise NotImplementedError(avx_no_tile_message)

    def store_tile(self, *, areg : greg_base,
                   treg : treg_base,
                   dt : adt):
        raise NotImplementedError(avx_no_tile_message)

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
                     rpref=self.rpref,
                     has_fp16=False
                     )
        self.fmul = avx_fmul(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     rpref=self.rpref,
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
        preg = self.rpref(vreg)
        return self.asmwrap(f"vpxor {preg},{preg},{preg}")

    def vreg(self, reg_idx):
        return xmm_vreg(reg_idx)

    def load_vector_bcast1(self, *, areg : greg_base,
                          vreg : vreg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pv = self.rpref(self.xmm_to_ymm(vreg))
        return self.asmwrap(f"vbroadcast{suf} ({pa}),{pv}")

    def load_vector_bcast1_immoff(self, *, areg : greg_base, offset : int,
                               vreg : vreg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pv = self.rpref(self.xmm_to_ymm(vreg))
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
                     rpref=self.rpref,
                     has_fp16=False
                     )
        self.fmul = avx_fmul(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     rpref=self.rpref,
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
        preg = self.rpref(vreg)
        return self.asmwrap(f"vpxor {preg},{preg},{preg}")

    def vreg(self, reg_idx : int):
        return ymm_vreg(reg_idx)

    def load_vector_bcast1(self, *, areg : greg_base,
                          vreg : vreg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pv = self.rpref(vreg)
        return self.asmwrap(f"vbroadcast{suf} ({pa}),{pv}")

    def load_vector_bcast1_immoff(self, *, areg : greg_base, offset : int,
                               vreg : vreg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pv = self.rpref(vreg)
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
                     rpref=self.rpref,
                     has_fp16=True
                     )
        self.fmul = avx_fmul(
                     asmwrap=self.asmwrap,
                     dt_suffixes=self.dt_suffixes,
                     it_suffixes=self.it_suffixes,
                     rpref=self.rpref,
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


    def isaquirks(self, *, rt : reg_tracker, dt : adt):
        maskreg = '%k2'
        if self.output_inline:
            maskreg = '%%k2'

        asmblock = super().isaquirks(rt=rt,dt=dt)
        return asmblock + self.asmwrap(f"kxnorb {maskreg},{maskreg},{maskreg}")

    def zero_vreg(self, *, vreg : vreg_base, dt : adt):
        preg = self.rpref(vreg)
        return self.asmwrap(f"vpxorq {preg},{preg},{preg}")

    def vreg(self, reg_idx : int):
        return zmm_vreg(reg_idx)

    def load_vector_bcast1(self, *, areg : greg_base,
                          vreg : vreg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pv = self.rpref(vreg)
        return self.asmwrap(f"vbroadcast{suf} ({pa}),{pv}")

    def load_vector_bcast1_immoff(self, *, areg : greg_base, offset : int,
                               vreg : vreg_base, dt : adt):
        suf = 's'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pv = self.rpref(vreg)
        return self.asmwrap(f"vbroadcast{suf} {offset}({pa}),{pv}")

    def load_vector_gather(self, *, areg : greg_base, offvreg : vreg_base,
                           vreg : vreg_base, dt : adt,
                           it : ait):
        suf = 'p'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pv = self.rpref(vreg)
        pov = self.rpref(offvreg)
        address = f"({pa},{pov},1)" # TODO: Explore using scale param
        isuf = self.it_suffixes[it]
        # TODO: properly implement mask register handling
        maskreg = '%k2'
        if self.output_inline:
            maskreg = '%%k2'
        return self.asmwrap(f"vgather{isuf}{suf} {address},{pv}{{{maskreg}}}")

    def store_vector_scatter(self, *, areg : greg_base, offvreg : vreg_base,
                           vreg : vreg_base, dt : adt,
                           it : ait):
        suf = 'p'+self.dt_suffixes[dt]
        pa = self.rpref(areg)
        pv = self.rpref(vreg)
        pov = self.rpref(offvreg)
        address = f"({pa},{pov},1)" # TODO: Explore using scale param
        isuf = self.it_suffixes[it]
        # TODO: properly implement mask register handling
        maskreg = '%k2'
        if self.output_inline:
            maskreg = '%%k2'
        return self.asmwrap(f"vscatter{isuf}{suf} {pv},{address}{{{maskreg}}}")
