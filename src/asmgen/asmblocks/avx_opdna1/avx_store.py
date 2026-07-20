# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
AVX/FMA/AVX2/AVX512 store instructions
"""

from typing import Callable,Any

from ...registers import asm_data_type as adt
from ..op import opdna1_action as action
from ..op.opdna1 import opdna1_modifier as mod
from .avx_opdna1_base import avx_opdna1,avx128_opdna1,avx256_opdna1,avx512_opdna1

class no_bcast(avx_opdna1):
    """
    Helper class for diagnosing lack of BCAST support
    """
    def diagnose_failure(self, modifiers : set[mod],
                         kwargs : dict[str,Any],
                         dts : dict[str, adt]):
        super().diagnose_failure(modifiers,kwargs,dts)

        if mod.BCAST in modifiers:
            raise ValueError("BCAST modifier can't be used with stores")

class no_vindex(avx_opdna1):
    """
    Helper class for diagnosing lack of VINDEX support
    """
    def diagnose_failure(self, modifiers : set[mod],
                         kwargs : dict[str,Any],
                         dts : dict[str, adt]):
        super().diagnose_failure(modifiers,kwargs,dts)

        if mod.VINDEX in modifiers:
            raise ValueError("VINDEX modifier can't be used with avx2 128/256 bit stores")

class avx128_store(avx128_opdna1,no_bcast,no_vindex):
    """
    AVX2 128bit stores
    """

    def __init__(self,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action.STORE, asmwrap=asmwrap, rpref=rpref)


# it's fine
# pylint: disable-next=too-many-ancestors
class avx256_store(avx256_opdna1,no_bcast,no_vindex):
    """
    AVX2 256bit stores
    """

    def __init__(self,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action.STORE, asmwrap=asmwrap, rpref=rpref)

class avx512_store(avx512_opdna1,no_bcast):
    """
    AVX512 stores
    """

    def __init__(self,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action.STORE, asmwrap=asmwrap, rpref=rpref)
