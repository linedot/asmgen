# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
SVE fadd instruction
"""

from typing import Any

from ...registers import asm_data_type as adt
from ..op import opd3_modifier as mod

from .sve_opd3_base import sve_opd3_base

class sve_fadd(sve_opd3_base):
    """
    SVE implementation of fadd
    """

    inst_base = "add"

    def diagnose_failure(self, modifiers : set[mod],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):
        super().diagnose_failure(modifiers, kwargs, dts)
        if mod.NP in modifiers:
            raise ValueError("NEON add has no NP-form")
