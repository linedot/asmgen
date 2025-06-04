# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RVV 1.0 and 0.7.1 fused-multiply-accumulate
"""

from typing import Callable

from ..operations import modifier

from .rvv_opd3_base import rvv_opd3_base

class rvv_fma(rvv_opd3_base):
    """
    RVV 1.0 and 0.7.1 implementation of fma
    """

    def __init__(self,
                 asmwrap : Callable[[str],str]):
        super().__init__(asmwrap=asmwrap)

        self.operand_order = [2,1,0]

    def get_base_inst(self, modifiers : set[modifier]):
        return "nmsac" if modifier.NP in modifiers else "macc"
