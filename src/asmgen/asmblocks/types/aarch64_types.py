# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
ARM64/AArch64 register types
"""
from ...registers import greg_base, freg_base, asm_data_type as adt

#pylint: disable=too-few-public-methods
class aarch64_greg(greg_base):
    """
    ARM64/AArch64 general purpose register
    """
    def __init__(self, reg_idx : int):
        self.reg_str = f"x{reg_idx}"
        if reg_idx == 31:
            self.reg_str = "sp"
        self.idx = reg_idx

    def get_wreg(self):
        if self.idx == 31:
            return "sp"
        return f"w{self.idx}"

    def __str__(self) -> str:
        return self.reg_str

class aarch64_freg(freg_base):
    """
    ARM64/AArch64 FP/scalar register
    """
    dt_regname_map : dict[adt,str] = {
        adt.FP8E4M3  : 'b',
        adt.FP8E5M2  : 'b',
        adt.FP16     : 'h',
        adt.FP32     : 's',
        adt.FP64     : 'd',
        adt.FP128    : 'q',
    }
    def __init__(self, reg_idx : int, dt : adt):
        self.idx = reg_idx
        self.dt = dt

    def __str__(self) -> str:
        return f"{self.dt_regname_map[self.dt]}{self.idx}"
