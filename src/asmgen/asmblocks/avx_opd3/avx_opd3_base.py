# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
AVX opd3 base case
"""
from abc import abstractmethod
from typing import Callable

from ...registers import (
    asm_data_type as adt,
    adt_triple,
    asm_index_type as ait,
    data_reg,
)
from ..operations import (
    opd3,
    widening_method,
    opd3_modifier as mod,
    operand_restriction
)

from ..types.avx_types import reg_prefixer, avx_vreg

from ...util import NIE_MESSAGE

class avx_opd3_base(opd3):
    """
    AVX base opd3 implementation with methods shared by all
    opd3 operations
    """

    # pylint: disable-next=too-many-positional-arguments
    def __init__(self,
                 asmwrap : Callable[[str],str],
                 dt_suffixes : dict[adt,str],
                 it_suffixes : dict[ait,str],
                 rpref : reg_prefixer,
                 has_fp16 : bool = False,
                 ):
        self.asmwrap = asmwrap
        self.dt_suffixes = dt_suffixes
        self.it_suffixes = it_suffixes
        self.rpref = rpref
        self.has_fp16 = has_fp16

    @abstractmethod
    def get_base_inst(self, modifiers : set[mod]) -> str:
        """
        Return the base instruction name based on the specified modifiers

        :param modifiers: set of modifiers to check the name for
        :type modifiers: set[class:`asmgen.asmblocks.operations.opd3_modifier`]
        :return: ASM instruction name
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @property
    def widening_method(self) -> widening_method:
        return widening_method.NONE

    def check_modifiers(self, modifiers : set[mod]):
        if mod.VF in modifiers:
            raise ValueError("AVX has no vf form")
        if mod.REGIDX in modifiers:
            raise ValueError("AVX has no regidx form")
        if mod.IDX in modifiers:
            raise ValueError("AVX has no idx form")
        if mod.PART in modifiers:
            raise ValueError("AVX has no partial instructions")
        if mod.MASK in modifiers:
            raise NotImplementedError("AVX masked opd3 not yet implemented")

    def supported_dts(self) -> list[dict[str,adt]]:
        supported_list = [
            {'adreg':adt.FP64, 'bdreg':adt.FP64, 'cdreg':adt.FP64},
            {'adreg':adt.FP32, 'bdreg':adt.FP32, 'cdreg':adt.FP32},
        ]
        if self.has_fp16:
            supported_list.append(
                    {'adreg':adt.FP16, 'bdreg':adt.FP16, 'cdreg':adt.FP16})
        return supported_list

    def get_required_params(self, modifiers: set[mod]) -> list[set[str]]:

        required_extra_params = []
        return required_extra_params

    def get_operand_restrictions(self, oprnd : str) -> set[operand_restriction]:
        # No restriction on any operands
        return {}

    def get_operand_restriction_value(self, oprnd : str,
                                      modifiers : set[mod],
                                      rstr : operand_restriction) \
      -> int|set[int]|tuple[str,int]:
        raise ValueError("No restriction {rstr} on operand {op} for AVX opd3")

    # modfier set is only read, therefore a mutable default is ok
    # pylint: disable-next=dangerous-default-value
    def implementation(self, *,
                       adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                     a_dt : adt, b_dt : adt, c_dt : adt,
                     modifiers : set[mod] = set(),
                     **kwargs) -> str:

        if any(not isinstance(r, avx_vreg) for r in (adreg,bdreg,cdreg)):
            raise ValueError("All dregs of an AVX opd3 must be avx_vreg")

        inst = self.get_base_inst(modifiers=modifiers)

        suf = 'p'+self.dt_suffixes[c_dt]
        pa = self.rpref(adreg)
        pb = self.rpref(bdreg)
        pc = self.rpref(cdreg)
        return self.asmwrap(f"{inst}{suf} {pa},{pb},{pc}")
