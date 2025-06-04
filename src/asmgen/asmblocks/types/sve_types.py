# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
SVE register types
"""
from ...registers import vreg_base

#pylint: disable=too-few-public-methods
class sve_vreg(vreg_base):
    """
    SVE vector register
    """
    def __init__(self, reg_idx : int):
        self.reg_str = f"z{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str

class sve_preg:
    """
    SVE predicate register
    """
    def __init__(self, reg_idx : int):
        self.reg_str = f"p{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str
