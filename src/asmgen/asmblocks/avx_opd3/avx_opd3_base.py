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
    data_reg
)
from ..operations import opd3,widening_method,modifier

from ..types.avx_types import prefix_if_raw_reg

from ...util import NIE_MESSAGE

class avx_opd3_base(opd3):
    """
    AVX base opd3 implementation with methods shared by all
    opd3 operations
    """

    def __init__(self,
                 asmwrap : Callable[[str],str],
                 dt_suffixes : dict[adt,str],
                 it_suffixes : dict[ait,str],
                 has_fp16 : bool = False
                 ):
        self.asmwrap = asmwrap
        self.dt_suffixes = dt_suffixes
        self.it_suffixes = it_suffixes
        self.has_fp16 = has_fp16

    @abstractmethod
    def get_base_inst(self, modifiers : set[modifier]) -> str:
        """
        Return the base instruction name based on the specified modifiers

        :param modifiers: set of modifiers to check the name for
        :type modifiers: set[class:`asmgen.asmblocks.operations.modifier`]
        :return: ASM instruction name
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @property
    def widening_method(self) -> widening_method:
        return widening_method.NONE

    def check_modifiers(self, modifiers : set[modifier]):
        if modifier.VF in modifiers:
            raise ValueError("AVX has no vf form")
        if modifier.REGIDX in modifiers:
            raise ValueError("AVX has no regidx form")
        if modifier.IDX in modifiers:
            raise ValueError("AVX has no idx form")
        if modifier.PART in modifiers:
            raise ValueError("AVX has no partial instructions")

    def supported_triples(self) -> list[adt_triple]:
        supported_list = [
            adt_triple(a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64),
            adt_triple(a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32),
        ]
        if self.has_fp16:
            supported_list.append(
                    adt_triple(a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16))
        return supported_list

    # modfier set is only read, therefore a mutable default is ok
    # pylint: disable-next=dangerous-default-value
    def __call__(self, *, adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                 a_dt : adt, b_dt : adt, c_dt : adt,
                 modifiers : set[modifier] = set(),
                 **kwargs) -> str:

        self.check_triple(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)
        if (a_dt != b_dt) or (a_dt != c_dt):
            raise ValueError("A,B and C must have same type")

        inst = self.get_base_inst(modifiers=modifiers)

        suf = 'p'+self.dt_suffixes[c_dt]
        pa = prefix_if_raw_reg(adreg)
        pb = prefix_if_raw_reg(bdreg)
        pc = prefix_if_raw_reg(cdreg)
        return self.asmwrap(f"{inst}{suf} {pa},{pb},{pc}")
