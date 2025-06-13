# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
NEON/ASIMD asm generator and related types
"""
from ..registers import (
    reg_tracker,
    asm_data_type as adt,
    adt_size,
    asm_index_type as ait,
    treg_base,vreg_base,freg_base,greg_base
)
from .aarch64 import aarch64

from .types.aarch64_types import aarch64_freg
from .types.neon_types import neon_vreg
from .neon_opd3 import neon_fma,neon_fmul

class neon(aarch64):
    """
    NEON/ASIMD implementation of asmgen
    """

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
        super().__init__()
        self.fma = neon_fma(asmwrap=self.asmwrap,
                            dt_suffixes=self.dt_suffixes,
                            dt_idxsuffixes=self.dt_idxsuffixes)
        self.fmul = neon_fmul(asmwrap=self.asmwrap,
                              dt_suffixes=self.dt_suffixes,
                              dt_idxsuffixes=self.dt_idxsuffixes)

    def get_req_flags(self) -> list[str]:
        """
        Return required flags in cpuinfo for this generator to be supported
        """
        return ['asimd']

    def supportedby_cpuinfo(self, cpuinfo : str) -> bool:
        req_flags = self.get_req_flags()
        supported = True
        for r in req_flags:
            if -1 == cpuinfo.find(r):
                supported = False
                break
        return supported

    def isaquirks(self, *, rt : reg_tracker, dt : adt):
        return ""
    def isaendquirks(self, *, rt : reg_tracker, dt : adt):
        return ""

    def jvzero(self, *, vreg1 : vreg_base, freg : freg_base,
               vreg2 : vreg_base,
               greg : greg_base, label : str,
               dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        asmblock  = self.asmwrap(f"fmaxv {freg}, {vreg2}.{suf}")
        asmblock += self.asmwrap(f"fcmp {freg}, #0.0")
        asmblock += self.asmwrap(f"b.eq .{label}")
        return asmblock

    def vreg_to_qreg(self, vreg : neon_vreg) -> aarch64_freg:
        """
        Returns the AArch64 128 FP register register corresponding to
        the specified vector register

        :param vreg: NEON/ASIMD vector register
        :type vreg: class:`asmgen.asmblocks.types.neon_types.neon_vreg`
        :return: 128 bit FP reg
        :rtype: class:`asmgen.asmblocks.types.aarch64_types.aarch64_freg`
        """
        return aarch64_freg(reg_idx=vreg.idx, dt=adt.FP128)

    @property
    def is_vla(self) -> bool:
        return False

    def indexable_elements(self, dt : adt) -> int:
        return self.simd_size//adt_size(dt)

    @property
    def max_vregs(self):
        return 32

    @property
    def simd_size(self):
        return 16

    def simd_size_to_greg(self, *, reg: greg_base, dt: adt) -> str:
        return self.mov_greg_imm(reg=reg, imm=16//adt_size(dt))

    @property
    def c_simd_size_function(self):
        return f"size_t get_simd_size() {{ return {self.simd_size}; }}"

    def add_greg_voff(self, *, reg : greg_base, offset : int, dt : adt):
        byte_offset = self.simd_size*offset
        return self.asmwrap(f"add {reg},{reg},#{byte_offset}")

    def zero_vreg(self, *, vreg : vreg_base, dt : adt):
        suf = self.dt_suffixes[dt]
        zeroreg = f"{self.dt_greg_pfx[dt]}zr"
        return self.asmwrap(f"dup {vreg}.{suf},{zeroreg}")

    def vreg(self, reg_idx : int) -> neon_vreg:
        return neon_vreg(reg_idx)

    def qreg(self, idx : int) -> aarch64_freg:
        """
        Returns the AArch64 128 FP register register corresponding to
        the specified register id
        :param idx: register id
        :type idx: int
        :return: 128 bit FP reg
        :rtype: class:`asmgen.asmblocks.types.aarch64_types.aarch64_freg`
        """
        return aarch64_freg(reg_idx=idx, dt=adt.FP128)

    def min_load_immoff(self, dt : adt) -> int:
        _ = dt # explicitly unused
        return 0

    def max_load_immoff(self, dt : adt) -> int:
        return 4095*adt_size(dt)*2

    @property
    def min_load_voff(self) -> int:
        return 0

    @property
    def max_load_voff(self) -> int:
        return 4096//8

    @property
    def max_add_voff(self) -> int:
        return 4096//self.simd_size

    def load_vector(self, *, areg : greg_base,
                    vreg : vreg_base, dt : adt) -> str:
        if not isinstance(vreg, neon_vreg):
            raise ValueError(f"{vreg} is not a NEON vreg")
        qv = self.vreg_to_qreg(vreg)
        return self.asmwrap(f"ldr {qv}, [{areg}]")

    def load_vector_voff(self, *, areg : greg_base, voffset : int,
                         vreg : vreg_base, dt : adt) -> str:
        if not isinstance(vreg, neon_vreg):
            raise ValueError(f"{vreg} is not a NEON vreg")
        qv = self.vreg_to_qreg(vreg)
        return self.asmwrap(f"ldr {qv}, [{areg}, #{voffset*self.simd_size}]")

    def load_vector_immoff(self, *, areg : greg_base, offset : int,
                           vreg : vreg_base, dt : adt) -> str:
        if not isinstance(vreg, neon_vreg):
            raise ValueError(f"{vreg} is not a NEON vreg")
        qv = self.vreg_to_qreg(vreg)
        return self.asmwrap(f"ldr {qv}, [{areg}, #{offset}]")

    def load_vector_dist1(self, *, areg : greg_base,
                          vreg : vreg_base, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        return self.asmwrap(f"ld1r {{{vreg}.{suf}}}, [{areg}]")

    def load_vector_dist1_boff(self, *, areg : greg_base, offset : int,
                               vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError(
                ("load_vector_dist1_boff doesn't make sense with NEON,"
                 " use load_vector_dist1_inc or load_vector_voff + fma_idx instead"))

    def load_vector_dist1_inc(self, *, areg : greg_base, offset : int,
                              vreg : vreg_base, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        return self.asmwrap(f"ld1r {{{vreg}.{suf}}}, [{areg}], #{offset}")

    def store_vector_voff(self, *, areg : greg_base, voffset : int,
                          vreg : vreg_base, dt : adt) -> str:
        if not isinstance(vreg, neon_vreg):
            raise ValueError(f"{vreg} is not a NEON vreg")
        qv = self.vreg_to_qreg(vreg)
        return self.asmwrap(f"str {qv}, [{areg}, #{voffset*self.simd_size}]")

    def store_vector(self, *, areg : greg_base,
                     vreg : vreg_base, dt : adt) -> str:
        if not isinstance(vreg, neon_vreg):
            raise ValueError(f"{vreg} is not a NEON vreg")
        qv = self.vreg_to_qreg(vreg)
        return self.asmwrap(f"str {qv}, [{areg}]")

    def load_vector_immstride(self, *, areg : greg_base, byte_stride : int,
                    vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError("NEON has no load with immediate stride")

    def load_vector_gregstride(self, *, areg : greg_base, sreg : greg_base,
                    vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError("NEON has no load with scalar register stride")

    def load_vector_gather(self, *, areg : greg_base, offvreg : vreg_base,
                           vreg : vreg_base, dt : adt,
                           it : ait) -> str:
        raise NotImplementedError("NEON has no load with vector register stride")

    def store_vector_immstride(self, *, areg : greg_base, byte_stride : int,
                    vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError("NEON has no store with immediate stride")

    def store_vector_gregstride(self, *, areg : greg_base, sreg : greg_base,
                    vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError("NEON has no store with scalar register stride")

    def store_vector_scatter(self, *, areg : greg_base, offvreg : vreg_base,
                             vreg : vreg_base, dt : adt,
                             it : ait) -> str:
        raise NotImplementedError("NEON has no store with vector register stride")

    # Unsupported functionality:
    def max_tregs(self, dt : adt) -> int:
        return 0

    def treg(self, reg_idx : int, dt : adt) -> treg_base:
        raise NotImplementedError("NEON has no tiles, use SME")

    def zero_treg(self, *, treg : treg_base, dt : adt) -> str:
        raise NotImplementedError("NEON has no tiles, use SME")

    def load_tile(self, *, areg : greg_base,
                   treg : treg_base,
                   dt : adt) -> str:
        raise NotImplementedError("NEON has no tiles, use SME")

    def store_tile(self, *, areg : greg_base,
                   treg : treg_base,
                   dt : adt) -> str:
        raise NotImplementedError("NEON has no tiles, use SME")
