# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
AVX/FMA/AVX2/AVX512 store instructions
"""

from ..operations import opdna1_action as action
from .avx_opdna1_base import avx128_opdna1,avx256_opdna1,avx512_opdna1

class avx128_store(avx128_opdna1):
    """
    NEON freg and greg stores
    """

    def __init__(self,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action.STORE, asmwrap=asmwrap, rpref=rpref)

class avx256_store(avx256_opdna1):
    """
    NEON freg and greg stores
    """

    def __init__(self,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action.STORE, asmwrap=asmwrap, rpref=rpref)

class avx512_store(avx512_opdna1):
    """
    NEON freg and greg stores
    """

    def __init__(self,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action.STORE, asmwrap=asmwrap, rpref=rpref)
