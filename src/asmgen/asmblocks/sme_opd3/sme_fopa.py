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
        
        if mod.IDX in modifiers:
            raise ValueError("SME has no idx form")
        if mod.REGIDX in modifiers:
            raise ValueError("SME has no regidx form")
        if mod.VF in modifiers:
            raise ValueError("SME has no vf form")
        if mod.PART in modifiers:
            raise ValueError("SME has no partial instructions (widening instructions 'dot' neighbours)")
        if mod.MASK in modifiers:
            raise NotImplementedError("SME masked opd3 not implemented yet")

    @property
    def widening_method(self) -> widening_method:
        return widening_method.DOT_NEIGHBOURS

    def supported_dts(self) -> list[dict[str,adt]]:
        return [
            {'adreg':adt.FP64, 'bdreg':adt.FP64, 'cdreg':adt.FP64},
            {'adreg':adt.FP32, 'bdreg':adt.FP32, 'cdreg':adt.FP32},
            {'adreg':adt.FP16, 'bdreg':adt.FP16, 'cdreg':adt.FP16},

            {'adreg':adt.FP16, 'bdreg':adt.FP16, 'cdreg':adt.FP32},

            {'adreg':adt.FP8E5M2, 'bdreg':adt.FP8E5M2, 'cdreg':adt.FP32},
            {'adreg':adt.FP8E5M2, 'bdreg':adt.FP8E5M2, 'cdreg':adt.FP16},
            {'adreg':adt.FP8E4M3, 'bdreg':adt.FP8E4M3, 'cdreg':adt.FP32},
            {'adreg':adt.FP8E4M3, 'bdreg':adt.FP8E4M3, 'cdreg':adt.FP16},

            {'adreg':adt.FP8E5M2, 'bdreg':adt.FP8E5M2, 'cdreg':adt.FP32},
            {'adreg':adt.FP8E5M2, 'bdreg':adt.FP8E5M2, 'cdreg':adt.FP16},
            {'adreg':adt.FP8E4M3, 'bdreg':adt.FP8E4M3, 'cdreg':adt.FP32},
            {'adreg':adt.FP8E4M3, 'bdreg':adt.FP8E4M3, 'cdreg':adt.FP16},

            {'adreg':adt.SINT16, 'bdreg':adt.SINT16, 'cdreg':adt.SINT64},
            {'adreg':adt.SINT16, 'bdreg':adt.SINT16, 'cdreg':adt.SINT32},
            {'adreg':adt.SINT8, 'bdreg':adt.SINT8, 'cdreg':adt.SINT32},

            {'adreg':adt.UINT16, 'bdreg':adt.UINT16, 'cdreg':adt.UINT64},
            {'adreg':adt.UINT16, 'bdreg':adt.UINT16, 'cdreg':adt.UINT32},
            {'adreg':adt.UINT8, 'bdreg':adt.UINT8, 'cdreg':adt.UINT32},

            {'adreg':adt.SINT16, 'bdreg':adt.UINT16, 'cdreg':adt.SINT64},
            {'adreg':adt.SINT16, 'bdreg':adt.UINT16, 'cdreg':adt.SINT32},
            {'adreg':adt.SINT8, 'bdreg':adt.UINT8, 'cdreg':adt.SINT32},

            {'adreg':adt.UINT16, 'bdreg':adt.SINT16, 'cdreg':adt.SINT64},
            {'adreg':adt.UINT16, 'bdreg':adt.SINT16, 'cdreg':adt.SINT32},
            {'adreg':adt.UINT8, 'bdreg':adt.SINT8, 'cdreg':adt.SINT32},
        ]

    def get_required_params(self, modifiers: set[mod]) -> list[set[str]]:

        required_extra_params = []

        return required_extra_params

    def get_operand_restrictions(self, oprnd : str) -> set[operand_restriction]:
        # No restriction on any operands
        return {}

    def get_operand_restriction_value(self, op : str,
                                      modifiers : set[mod],
                                      rstr : operand_restriction) \
      -> int|set[int]|tuple[str,int]:
        raise ValueError("No restriction {rstr} on operand {op} for SME opd3")

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
    def implementation(self, *,
                       adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                       a_dt : adt, b_dt : adt, c_dt : adt,
                       modifiers : set[mod] = set(),
                       **kwargs) -> str:

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
