# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
AVX fadd instruction
"""

from typing import Any

from ..op import opd3_modifier as mod
from ...registers import asm_data_type as adt

from .avx_opd3_base import avx_opd3_base

class avx_fadd(avx_opd3_base):
    """
    AVX implementation of fadd
    """

    def get_base_inst(self, modifiers : set[mod]):
        return "vadd"

    def diagnose_failure(self, modifiers: set[mod],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):
        super().diagnose_failure(modifiers, kwargs, dts)
        if mod.NP in modifiers:
            raise ValueError("AVX fadd has no NP form")
