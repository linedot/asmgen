# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RVV 1.0 and 0.7.1 multiplication
"""

from ..operations import modifier

from .rvv_opd3_base import rvv_opd3_base

class rvv_fmul(rvv_opd3_base):
    """
    RVV 1.0 and 0.7.1 implementation of fma
    """

    def get_base_inst(self, modifiers : set[modifier]):
        return "mul"
