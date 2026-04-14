# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
AVX fadd instruction
"""

from ..operations import modifier

from .avx_opd3_base import avx_opd3_base

class avx_fadd(avx_opd3_base):
    """
    AVX implementation of fadd
    """

    def get_base_inst(self, modifiers):
        return "vadd"

    def check_modifiers(self, modifiers : set[modifier]):
        super().check_modifiers(modifiers=modifiers)
        if modifier.NP in modifiers:
            raise ValueError("AVX fadd has no NP form")
