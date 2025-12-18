# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
X86/AVX register types
"""
from typing import Callable

from ...registers import vreg_base, greg_base, freg_base, data_reg

#pylint: disable=too-few-public-methods
class x86_greg(greg_base):
    """
    x86_64 general purpose register
    """

    greg_names = [f'{i}' for i in \
            [str(j) for j in range(8,16)]+\
            ['a','b','c','d','si','di','bp','sp']]

    def __init__(self, reg_idx : int):
        self.reg_idx = reg_idx

    def name(self, size : int = 8):
        prefixes = {1:'', 2: '', 4: 'e', 8: 'r'}

        num_suffixes = {1:'b', 2: 'w', 4: 'd', 8: ''}

        alpha_suffixes = {1:'l', 2: '', 4: '', 8: ''}

        name = x86_greg.greg_names[self.reg_idx]

        # a,b,c,d
        if self.reg_idx in [8,9,10,11]:
            if size > 1:
                name += 'x'

        if self.reg_idx < 8:
            name = 'r' + name + num_suffixes[size]
        else:
            name = prefixes[size] + name + alpha_suffixes[size]

        return name

    def __str__(self) -> str:
        return self.name()

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


class reg_prefixer:
    """
    Helper class for prefixing a register with '%%' in inline ASM and '%' in normal ASM

    This ensures the registers are refered to correctly in AT&T/GAS style syntax
    """

    def __init__(self, output_inline_getter : Callable[[],bool]):
        self.output_inline_getter = output_inline_getter

    @property
    def output_inline(self) -> bool:
        return self.output_inline_getter()

    def __call__(self, reg : data_reg|greg_base, size : int = 8) -> str:
        """
        Depending on inline output state, prepends the string representation of the
        register with a '%%' for inline ASM and '%' for normal ASM
        """

        if not isinstance(reg, (x86_greg, avx_freg, xmm_vreg, ymm_vreg, zmm_vreg)):
            raise ValueError(f"{reg} is not a x86 or AVX register")
        regstr = str(reg)
        if isinstance(reg, x86_greg):
            regstr = reg.name(size)
        # If there is a [ it's probably a parameter
        if '[' in regstr:
            return regstr
        pref = '%%' if self.output_inline else '%'
        return f"{pref}{regstr}"
