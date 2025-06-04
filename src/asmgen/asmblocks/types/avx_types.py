# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
X86/AVX register types
"""
from typing import Union

from ...registers import vreg_base, greg_base, freg_base, data_reg

#pylint: disable=too-few-public-methods
class x86_greg(greg_base):
    """
    x86_64 general purpose register
    """

    greg_names = [f'r{i}' for i in \
            [str(j) for j in range(8,16)]+\
            ['ax','bx','cx','dx','si','di','bp','sp']]

    def __init__(self, reg_idx : int):
        self.reg_str = x86_greg.greg_names[reg_idx]

    def __str__(self) -> str:
        return self.reg_str

class avx_freg(freg_base):
    """
    x86_64 scalar register (actually xmm)
    """
    def __init__(self, reg_idx : int):
        self.reg_str = f"xmm{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str

class xmm_vreg(vreg_base):
    """
    AVX 128 bit vector register (xmm)
    """
    def __init__(self, reg_idx : int):
        self.reg_str = f"xmm{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str

class ymm_vreg(vreg_base):
    """
    AVX 256 bit vector register (ymm)
    """
    def __init__(self, reg_idx : int):
        self.reg_str = f"ymm{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str

class zmm_vreg(vreg_base):
    """
    AVX512 vector register (zmm)
    """
    def __init__(self, reg_idx : int):
        self.reg_str = f"zmm{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str

def prefix_if_raw_reg(reg : Union[data_reg,greg_base]) -> str:
    """
    Prepends '%%' to register names for AT&T/GAS style ASM
    """
    if not isinstance(reg, (x86_greg, avx_freg, xmm_vreg, ymm_vreg, zmm_vreg)):
        raise ValueError(f"{reg} is not a x86 or AVX register")
    # If there is a [ it's probably a parameter
    if '[' in str(reg):
        return str(reg)
    return f"%%{reg}"
