# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
SME fopa implementation
"""

from typing import Callable

from ...registers import (
    asm_data_type as adt,
    adt_size,
    adt_triple,
    adt_is_float,
    adt_is_int,
    adt_is_signed,
    adt_is_unsigned,
    data_reg
)
from ..operations import opd3, opd3_modifier as mod
from ..types.sme_types import sme_treg
from ..types.sve_types import sve_vreg

class sme_fopa(opd3):
    """
    SME ASM instructions of fused outer-product-accumulate operations
    """

    NIE_MESSAGE = "Not supported in SME"

    def __init__(self,
                 asmwrap : Callable[[str],str],
                 dt_suffixes : dict[adt,str]):
        self.asmwrap = asmwrap
        self.dt_suffixes = dt_suffixes

    def check_modifiers(self, modifiers : set[mod]):
        if modifiers.intersection(
                set([mod.IDX, mod.PART, mod.VF])):
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
                 modifiers : set[mod] = set(),
                 **kwargs) -> str:
        self.check_triple(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)
        self.check_modifiers(modifiers=modifiers)

        if not isinstance(cdreg, sme_treg):
            raise ValueError(f"{cdreg} is not an sme_treg")

        if not isinstance(adreg, sve_vreg):
            raise ValueError(f"{adreg} is not an sve_vreg")
        if not isinstance(bdreg, sve_vreg):
            raise ValueError(f"{bdreg} is not an sve_vreg")


        suf = "s" if mod.NP in modifiers else "a"
        inst = self.mopx_inst_str(a_dt=a_dt, b_dt=b_dt, suf=suf)
        narrow_suf = self.dt_suffixes[a_dt]
        wide_suf = self.dt_suffixes[c_dt]
        return self.asmwrap(
            f"{inst} {cdreg}.{wide_suf},p0/m,p0/m,{adreg}.{narrow_suf},{bdreg}.{narrow_suf}")
