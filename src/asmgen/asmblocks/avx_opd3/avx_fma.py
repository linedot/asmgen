# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
AVX fma instruction
"""

from ..operations import modifier

from .avx_opd3_base import avx_opd3_base

class avx_fma(avx_opd3_base):
    """
    AVX implementation of fma
    """

    def get_base_inst(self, modifiers):
        return "vfnmadd231" if modifier.NP in modifiers else "vfmadd231"
