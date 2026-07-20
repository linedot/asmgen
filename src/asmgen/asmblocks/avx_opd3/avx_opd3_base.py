# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
AVX opd3 base case
"""
from abc import abstractmethod
from typing import Callable,Any

from ...registers import (
    asm_data_type as adt,
    asm_index_type as ait,
    data_reg,
)
from ..op import (
    opd3,
    opd3_modifier as mod,
    operation_signature
)

from ..types.avx_types import reg_prefixer

from ...util import NIE_MESSAGE

from .signatures import make_avx_opd3_signatures

class avx_opd3_base(opd3):
    """
    AVX base opd3 implementation with methods shared by all
    opd3 operations
    """

    supports_np = False

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

        self.signatures = make_avx_opd3_signatures(supports_np=self.supports_np,
                                                   has_fp16=self.has_fp16)

    def get_signatures(self) -> list[operation_signature]:
        return self.signatures

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

    def diagnose_failure(self, modifiers: set[mod],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):
        if mod.VF in modifiers:
            raise ValueError("AVX has no vf form")
        if mod.REGIDX in modifiers:
            raise ValueError("AVX has no regidx form")
        if mod.IDX in modifiers:
            raise ValueError("AVX has no idx form")
        if mod.BLOCKIDX in modifiers:
            raise ValueError("AVX has no block-idx form")
        if mod.PART in modifiers:
            raise ValueError("AVX has no partial instructions")
        if mod.MASK in modifiers:
            raise NotImplementedError("AVX masked opd3 not yet implemented")

    def implementation(self, *,
                       adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                       a_dt : adt, b_dt : adt, c_dt : adt,
                       modifiers : set[mod] = None,
                       **kwargs) -> str:

        if modifiers is None:
            modifiers = set()

        inst = self.get_base_inst(modifiers=modifiers)

        suf = 'p'+self.dt_suffixes[c_dt]
        pa = self.rpref(adreg)
        pb = self.rpref(bdreg)
        pc = self.rpref(cdreg)
        return self.asmwrap(f"{inst}{suf} {pa},{pb},{pc}")
