# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Base AArch64 store instructions
"""

from typing import Callable

from ..op import opdna1_action as action
from .aarch64_opdna1_base import aarch64_opdna1

class aarch64_store(aarch64_opdna1):
    """
    RISC-V freg and greg stores
    """

    def __init__(self, asmwrap : Callable[[str],str]):
        super().__init__(action=action.STORE, asmwrap=asmwrap)
