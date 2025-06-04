# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RISC-V 64-bit register types
"""
from ...registers import greg_base, freg_base

#pylint: disable=too-few-public-methods

class riscv64_greg(greg_base):
    """
    RISC-V 64bit general purpose register
    """

    # according to calling convention: temporaries, saved, function arguments,
    # leave the sp,gp, return address, etc... alone
    names = [f't{i}' for i in range(7)] +\
            [f's{i}' for i in range(1,12)] +\
            [f'a{i}' for i in range(8)]

    def __init__(self, reg_idx : int):
        self.reg_str = riscv64_greg.names[reg_idx]

    def __str__(self) -> str:
        return self.reg_str

class riscv64_freg(freg_base):
    """
    RISC-V 64bit scalar register
    """

    names = [f'f{i}' for i in range(32)]

    def __init__(self, reg_idx : int):
        self.reg_str = riscv64_freg.names[reg_idx]

    def __str__(self) -> str:
        return self.reg_str
