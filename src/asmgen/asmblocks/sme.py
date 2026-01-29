# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
SME asm generator
"""

from typing import Callable

from ..registers import (
    asm_data_type as adt,
    reg_tracker,
    adt_is_float,
    adt_is_int,
    adt_triple,
    adt_size,
    data_reg,
    treg_base, vreg_base, greg_base,
)
from .sve import sve
from .operations import opd3,widening_method,modifier


# pylint: disable-next=too-few-public-methods
class sme_treg(treg_base):
    """
    SME tile register
    """
    def __init__(self, reg_idx : int, dt : adt):
        # FP64,I64 : 8
        # FP32,I32 : 4
        # FP16,I16 : 2
        # FP8,I8 : 1
        max_tiles = adt_size(dt)

        if reg_idx > max_tiles:
            raise ValueError(f"SME has no tile {reg_idx} for data type {dt}")

        self.reg_str = f"za{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str

class sme_fmopa(opd3):
    """
    SME ASM instructions of fused outer-product-accumulate operations
    """

    NIE_MESSAGE = "Not supported in SME"

    def __init__(self,
                 asmwrap : Callable[[str],str],
                 dt_suffixes : dict[adt,str]):
        self.asmwrap = asmwrap
        self.dt_suffixes = dt_suffixes

    def check_modifiers(self, modifiers : set[modifier]):
        if modifiers.intersection(
                set([modifier.IDX, modifier.PART, modifier.VF])):
            raise ValueError("unsupported modifiers for SME")

    @property
    def widening_method(self) -> widening_method:
        return widening_method.DOT_NEIGHBOURS

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
        """
        Choose the correct base MOPX instruction based on specified types

        :param a_dt: Type of the A component
        :type a_dt: class:`asmgen.registers.asm_data_type`
        :param b_dt: Type of the B component
        :type b_dt: class:`asmgen.registers.asm_data_type`
        :param suf: mop suffix (accumulate, subtract,...)
        :type suf: str
        :return: string containing the base instruction
        :rtype: str
        """
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

        raise ValueError("Unsupported datatypes a={a_dt},b={b_dt}")

    # modfier set is only read, therefore a mutable default is ok
    # pylint: disable-next=dangerous-default-value
    def __call__(self, *, adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                 a_dt : adt, b_dt : adt, c_dt : adt,
                 modifiers : set[modifier] = set(),
                 **kwargs) -> str:
        self.check_triple(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)
        self.check_modifiers(modifiers=modifiers)

        if a_dt != b_dt:
            raise ValueError("A and B must have same type")
        if adt_size(a_dt) > adt_size(c_dt):
            raise ValueError("C type can't have smaller size than A/B type")
        if (adt_is_float(c_dt) and adt_is_int(a_dt)) or\
           (adt_is_float(a_dt) and adt_is_int(c_dt)):
            raise ValueError("Accumulator and multiplicands must be both either fp or int types")
        valid_c_types = [adt.FP64, adt.FP32, adt.FP16,
                         adt.UINT64, adt.UINT32, adt.UINT16,
                         adt.SINT64, adt.SINT32, adt.SINT16]
        if c_dt not in valid_c_types:
            valid_str = ','.join([str(t) for t in valid_c_types])
            raise ValueError(f"C type must be one of [{valid_str}]")


        suf = "s" if modifier.NP in modifiers else "a"
        inst = self.mopx_inst_str(a_dt=a_dt, b_dt=b_dt, suf=suf)
        narrow_suf = self.dt_suffixes[a_dt]
        wide_suf = self.dt_suffixes[c_dt]
        return self.asmwrap(
            f"{inst} {cdreg}.{wide_suf},p0/m,p0/m,{adreg}.{narrow_suf},{bdreg}.{narrow_suf}")

class sme(sve):
    """
    SME asmgen implementation
    """

    def __init__(self):
        super().__init__()
        self.valid_rcregs = [str(self.greg(i)) for i in range(12,16)]
        self.fopa = sme_fmopa(asmwrap=self.asmwrap,
                              dt_suffixes=self.dt_suffixes)

    @property
    def c_simd_size_function(self):
        pre_oi = self.output_inline
        self.set_output_inline(yesno=True)
        result  = "inline size_t get_simd_size() {\n"
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
        self.set_output_inline(yesno=pre_oi)
        return result

    def isaquirks(self, *, rt : reg_tracker, dt : adt) -> str:
        asmblock = self.asmwrap("smstart")
        asmblock += super().isaquirks(rt=rt,dt=dt)
        return asmblock

    def isaendquirks(self, *, rt : reg_tracker, dt : adt) -> str:
        asmblock = super().isaendquirks(rt=rt,dt=dt)
        asmblock += self.asmwrap("smstop")
        return asmblock

    def max_tregs(self, dt : adt) -> int:
        return adt_size(dt)

    def treg(self, reg_idx : int, dt : adt) -> treg_base:
        return sme_treg(reg_idx, dt)

    def zero_treg(self, *, treg : treg_base, dt : adt) -> str:
        suf = self.dt_suffixes[dt]
        return self.asmwrap(f"zero {treg}.{suf}")


    def insert_tile_rows(self, *,
                       rreg : greg_base,
                       roff_start : int, roff_end : int,
                       treg : treg_base,
                       vregs : list[vreg_base],
                       dt :adt) -> str:
        """
        Returns ASM string for inserting one or multiple vregs as rows into the treg

        :param rreg: register containing the starting row; must be in `sme.valid_rcregs`
        :type rreg: class:`asmgen.register.greg_base`
        :param roff_start: row offset to base row to start inserting at
        :type roff_start: int
        :param roff_end: row offset to base row to stop inserting at
        :type roff_end: int
        :param treg: tile register to insert rows into
        :type treg: class:`asmgen.register.treg_base`
        :param vregs: list of vector registers containing the rows to insert
        :type vregs: list[class:`asmgen.register.treg_base`]
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the necessary ASM instructions
        :rtype: str
        """
        suf = self.dt_suffixes[dt]
        if str(rreg) not in self.valid_rcregs:
            raise ValueError((f"rreg {rreg} is not a valid row/column offset register"
                              f"(allowed: {','.join(self.valid_rcregs)})"))

        vsrc = "{" + ", ".join([f"{vreg}.{suf}" for vreg in vregs]) + "}"

        opt_preg = ""
        roff_str = f"{roff_start}:{roff_end}"
        if 1 == len(vregs):
            opt_preg = ",p0/m"
            roff_str = f"{roff_start}"

        return self.asmwrap(
            f"mov {treg}h.{suf}[{rreg},{roff_str}]{opt_preg},{vsrc}")

    def extract_tile_rows(self, *,
                       rreg : greg_base,
                       roff_start : int, roff_end : int,
                       treg : treg_base,
                       vregs : list[vreg_base],
                       dt :adt):
        """
        Returns ASM string for extracting one or multiple rows from the treg into
        the specified vregs

        :param rreg: register containing the starting row; must be in `sme.valid_rcregs`
        :type rreg: class:`asmgen.register.greg_base`
        :param roff_start: row offset to base row to start extracting at
        :type roff_start: int
        :param roff_end: row offset to base row to stop extracting at
        :type roff_end: int
        :param treg: tile register to extract rows from
        :type treg: class:`asmgen.register.treg_base`
        :param vregs: list of vector registers to write the rows to
        :type vregs: list[class:`asmgen.register.treg_base`]
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the necessary ASM instructions
        :rtype: str
        """
        suf = self.dt_suffixes[dt]
        if str(rreg) not in self.valid_rcregs:
            raise ValueError((f"rreg {rreg} is not a valid row/column offset register"
                              f" (allowed: {','.join(self.valid_rcregs)})"))

        vdest = "{" + ", ".join([f"{vreg}.{suf}" for vreg in vregs]) + "}"

        opt_preg = ""
        roff_str = f"{roff_start}:{roff_end}"
        if 1 == len(vregs):
            opt_preg = ",p0/m"
            roff_str = f"{roff_start}"

        return self.asmwrap(
            f"mov {vdest}{opt_preg},{treg}h.{suf}[{rreg},{roff_str}]")


    def load_tile_row(self, *,
                       areg : greg_base,
                       rreg : greg_base,
                       roff : int,
                       voff : int,
                       treg : treg_base,
                       dt :adt) -> str:
        """
        Returns ASM string for loading a row of data into a treg

        :param areg: GP register containing the base memory address
        :type areg: class:`asmgen.register.greg_base`

        :param rreg: register containing the base row; must be in `sme.valid_rcregs`
        :type rreg: class:`asmgen.register.greg_base`
        :param roff: row offset to base row to load the data into
        :type roff: int
        :param voff: offset to base address in number of vectors
        :type voff: int
        :param treg: tile register to load the row into
        :type treg: class:`asmgen.register.treg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the necessary ASM instructions
        :rtype: str
        """
        if str(rreg) not in self.valid_rcregs:
            raise ValueError((f"rreg {rreg} is not a valid row/column offset register"

                              f" (allowed: {','.join(self.valid_rcregs)})"))

        rreg_str = str(rreg).replace('x','w')
        suf = self.dt_suffixes[dt]
        msuf = self.dt_mnem_suffixes[dt]
        address = f"[{areg}]"
        if voff > 0:
            address = f"[{areg}, #{voff}, MUL VL]"
        return self.asmwrap(
            f"ld1{msuf} {{{treg}h.{suf}[{rreg_str},{roff}]}},p0/z,{address}")

    def store_tile_row(self, *,
                       areg : greg_base,
                       rreg : greg_base,
                       roff : int,
                       voff : int,
                       treg : treg_base,
                       dt :adt):
        """
        Returns ASM string for storing a row of data from a treg into memory

        :param areg: GP register containing the base memory address
        :type areg: class:`asmgen.register.greg_base`

        :param rreg: register containing the base row; must be in `sme.valid_rcregs`
        :type rreg: class:`asmgen.register.greg_base`
        :param roff: row offset to base row to store the data from
        :type roff: int
        :param voff: offset to base address in number of vectors
        :type voff: int
        :param treg: tile register to store the row from
        :type treg: class:`asmgen.register.treg_base`
        :param dt: Data type of the values
        :type dt: class:`asmgen.registers.asm_data_type`
        :return: String containing the necessary ASM instructions
        :rtype: str
        """
        if str(rreg) not in self.valid_rcregs:
            raise ValueError((f"rreg {rreg} is not a valid row/column offset register"
                              f" (allowed: {','.join(self.valid_rcregs)})"))

        rreg_str = str(rreg).replace('x','w')
        suf = self.dt_suffixes[dt]
        msuf = self.dt_mnem_suffixes[dt]
        address = f"[{areg}]"
        if voff > 0:
            address = f"[{areg}, #{voff}, MUL VL]"
        return self.asmwrap(
            f"st1{msuf} {{{treg}h.{suf}[{rreg_str},{roff}]}},p0/m,{address}")

    def load_tile(self, *, areg : greg_base,
                   treg : treg_base, dt : adt) -> str:
        raise NotImplementedError(
            "SME has no tile loading instruction, use load_vector* and insert_tile_row/column")

    def store_tile(self, *, areg : greg_base,
                   treg : treg_base, dt : adt) -> str:
        raise NotImplementedError(
            "SME has no tile storing instruction, use extract_tile_row/column and store_vector*")
