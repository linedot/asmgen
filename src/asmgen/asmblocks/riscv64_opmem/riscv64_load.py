# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RISC-V +D/F load instructions
"""
from typing import Callable

from ..op import opmem_action as action
from .riscv64_opmem_base import riscv64_opmem

class riscv64_load(riscv64_opmem):
    """
    RISC-V freg and greg loads
    """

    def __init__(self, asmwrap : Callable[[str],str]):
        super().__init__(action=action.LOAD,asmwrap=asmwrap)
