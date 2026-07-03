# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
AVX fmul instruction
"""

from ..operations import opd3_modifier as mod

from .avx_opd3_base import avx_opd3_base

class avx_fmul(avx_opd3_base):
    """
    AVX implementation of fmul
    """

    def get_base_inst(self, modifiers : set[mod]):
        return "vmul"

    def check_modifiers(self, modifiers : set[mod]):
        super().check_modifiers(modifiers=modifiers)
        if mod.NP in modifiers:
            raise ValueError("AVX fmul has no NP form")
