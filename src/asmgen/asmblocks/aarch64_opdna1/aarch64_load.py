# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Base AArch64 load instructions
"""

from ..operations import opdna1_action as action
from .aarch64_opdna1_base import aarch64_opdna1

class aarch64_load(aarch64_opdna1):
    """
    RISC-V freg and greg loads
    """

    def __init__(self):
        super().__init__(action=action.LOAD)
