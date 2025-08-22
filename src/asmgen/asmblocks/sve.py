# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
SVE asm generator
"""

from .aarch64 import aarch64

from ..registers import (
    reg_tracker,
    asm_data_type as adt,
    adt_size,
    asm_index_type as ait,
    treg_base, vreg_base, freg_base, greg_base
)

from .types.sve_types import sve_vreg,sve_preg
from .sve_opd3 import sve_fma,sve_fmul

from .neon import neon

class sve(aarch64):
    """
    SVE asmgen implementation
    """

    dt_suffixes = neon.dt_idxsuffixes

    dt_mnem_suffixes = {
            adt.FP128  : "q",
            adt.DOUBLE : "d",
            adt.SINT64 : "d",
            adt.UINT64 : "d",
            adt.SINGLE : "w",
            adt.SINT32 : "w",
            adt.UINT32 : "w",
            adt.HALF   : "h",
            adt.SINT16 : "h",
            adt.UINT16 : "h",
            adt.FP8E4M3: "b",
            adt.FP8E5M2: "b",
            adt.UINT8  : "b",
            adt.SINT8  : "b",
            }

    def __init__(self):
        super().__init__()
        self.fma = sve_fma(asmwrap=self.asmwrap,
                           dt_suffixes=self.dt_suffixes,
                           dt_idxsuffixes=self.dt_suffixes)
        self.fmul = sve_fmul(asmwrap=self.asmwrap,
                           dt_suffixes=self.dt_suffixes,
                           dt_idxsuffixes=self.dt_suffixes)

    def get_req_flags(self) -> list[str]:
        """
        Return required flags in cpuinfo for this generator to be supported
        """
        return ['sve']

    def supportedby_cpuinfo(self, cpuinfo : str) -> bool:
        req_flags = self.get_req_flags()
        supported = True
        for r in req_flags:
            if -1 == cpuinfo.find(r):
                supported = False
                break
        return supported

    def isaquirks(self, *, rt : reg_tracker, dt : adt) -> str:
        asmblock = self.ptrue(self.preg(0), dt)
        return asmblock

    def isaendquirks(self, *, rt : reg_tracker, dt : adt) -> str:
        return ""

    def jvzero(self, *, vreg1 : vreg_base, freg : freg_base,
               vreg2 : vreg_base,
               greg : greg_base, label : str,
               dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        asmblock  = self.asmwrap(f"fcmne p1.d,{vreg1}.{suf},p0/z,#0,0")
        asmblock += self.asmwrap( "ptest p0, p1.b")
        asmblock += self.asmwrap(f"b.any .{label}")
        return asmblock


    @property
    def is_vla(self) -> bool:
        return True

    @property
    def max_vregs(self) -> int:
        return 32

    @property
    def simd_size(self) -> int:
        return 1


    def simd_size_to_greg(self, *, reg: greg_base, dt: adt) -> str:
        suf = self.dt_suffixes[dt]
        result  = self.asmwrap(f"mov {reg},#0")
        result += self.asmwrap(f"inc{suf} {reg}")

        return result

    def indexable_elements(self, dt : adt) -> int:
        # 128 bits are indexable
        return 16//adt_size(dt)

    @property
    def c_simd_size_function(self) -> str:
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

    def add_greg_voff(self, *, reg : greg_base, offset : int, dt : adt) -> str:
        return self.asmwrap(f"incb {reg}, ALL, MUL #{offset}")

    def zero_vreg(self, *, vreg : vreg_base, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        return self.asmwrap(f"dup {vreg}.{suf},#0")

    def vreg(self, reg_idx : int) -> sve_vreg:
        return sve_vreg(reg_idx)

    def min_load_immoff(self, dt : adt) -> int:
        return 0

    def max_load_immoff(self, dt : adt) -> int:
        return 252

    @property
    def min_load_voff(self) -> int:
        return -8

    @property
    def max_load_voff(self) -> int:
        return 7

    @property
    def max_add_voff(self) -> int:
        return 16

    def load_vector(self, *, areg : greg_base,
                    vreg : vreg_base, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        msuf = self.dt_mnem_suffixes[dt]
        return self.asmwrap(f"ld1{msuf} {vreg}.{suf}, p0/z, [{areg}]")

    def load_vector_voff(self, *, areg : greg_base, voffset : int,
                         vreg : vreg_base, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        msuf = self.dt_mnem_suffixes[dt]
        if voffset > self.max_load_voff:
            raise ValueError(f"voffset {voffset} > {self.max_load_voff}")
        return self.asmwrap(f"ld1{msuf} {vreg}.{suf}, p0/z, [{areg}, #{voffset}, MUL VL]")

    def load_vector_immoff(self, *, areg : greg_base, offset : int,
                           vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError("SVE has no vector loads with immediate offset, use load_vector_voff")

    def load_vector_bcast1(self, *, areg : greg_base,
                          vreg : vreg_base, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        msuf = self.dt_mnem_suffixes[dt]
        return self.asmwrap(f"ld1r{msuf} {vreg}.{suf}, p0/z, [{areg}]")

    def load_vector_bcast1_immoff(self, *, areg : greg_base, offset : int,
                               vreg : vreg_base, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        msuf = self.dt_mnem_suffixes[dt]
        return self.asmwrap(f"ld1r{msuf} {vreg}.{suf}, p0/z, [{areg}, #{offset}]")

    def load_vector_bcast1_inc(self, *, areg : greg_base, offset : int,
                              vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError(
                "SVE doesn't have a post-index ld1r{suf}, use load_vector_bcast1_immoff instead")

    def store_vector_voff(self, *, areg : greg_base, voffset : int,
                          vreg : vreg_base, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        msuf = self.dt_mnem_suffixes[dt]
        address = f"[{areg}, #{voffset}, MUL VL]"
        if 0 == voffset:
            address = f"[{areg}]"
        return self.asmwrap(f"st1{msuf} {{{vreg}.{suf}}}, p0, {address}")

    def store_vector(self, *, areg : greg_base,
                     vreg : vreg_base, dt : adt) -> str:
        return self.store_vector_voff(areg=areg, voffset=0, vreg=vreg, dt=dt)

    # SVE-specific
    def preg(self, idx : int) -> sve_preg:
        """
        returns an SVE predicate register with the specified index

        :param idx: Register index
        :type idx: int
        :return: Predicate register
        :rtype: class:`asmgen.asmblocks.types.sve_types.sve_preg`
        """
        return sve_preg(idx)

    def ptrue(self, reg : sve_preg, dt : adt) -> str:
        """
        Stores an all-true mask for the specified data type in the specified
        predicate register

        :param reg: Predicate register
        :type reg: class:`asmgen.asmblocks.types.sve_types.sve_preg`
        :param dt: Data type to use
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String with the required SVE ASM
        :rtype: str
        """
        suf = self.dt_suffixes[dt]
        return self.asmwrap(f"ptrue {reg}.{suf}")


    def load_vector_immstride(self, *, areg : greg_base, byte_stride : int,
                    vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError("SVE has no load with immediate stride")

    def load_vector_gregstride(self, *, areg : greg_base, sreg : greg_base,
                    vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError("SVE has no load with scalar register stride")

    def load_vector_gather(self, *, areg : greg_base, offvreg : vreg_base,
                           vreg : vreg_base, dt : adt,
                           it : ait) -> str:
        _ = it # explicitly unused
        suf = self.dt_suffixes[dt]
        msuf = self.dt_mnem_suffixes[dt]
        return self.asmwrap(f"ld1{msuf}.v {vreg}.{suf}, p0/z,[{areg}, {offvreg}]")

    def store_vector_immstride(self, *, areg : greg_base, byte_stride : int,
                    vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError("RVV has no store with immediate stride")

    def store_vector_gregstride(self, *, areg : greg_base, sreg : greg_base,
                    vreg : vreg_base, dt : adt) -> str:
        raise NotImplementedError("SVE has no store with scalar register stride")

    def store_vector_scatter(self, *, areg : greg_base, offvreg : vreg_base,
                             vreg : vreg_base, dt : adt,
                             it : ait) -> str:
        _ = it # explicitly unused
        suf = self.dt_suffixes[dt]
        msuf = self.dt_mnem_suffixes[dt]
        return self.asmwrap(f"st1{msuf}.v {vreg}.{suf}, p0, [{areg}, {offvreg}]")


    # Unsupported functionality:
    def max_tregs(self, dt : adt) -> int:
        return 0

    def treg(self, reg_idx : int, dt : adt) -> treg_base:
        raise NotImplementedError("SVE has no tiles, use SME")

    def zero_treg(self, *, treg : treg_base, dt : adt) -> str:
        raise NotImplementedError("SVE has no tiles, use SME")

    def load_tile(self, *, areg : greg_base,
                   treg : treg_base,
                   dt : adt) -> str:
        raise NotImplementedError("SVE has no tiles, use SME")

    def store_tile(self, *, areg : greg_base,
                   treg : treg_base,
                   dt : adt) -> str:
        raise NotImplementedError("SVE has no tiles, use SME")
