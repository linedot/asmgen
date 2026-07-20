# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Base X86 load instructions
"""

from typing import Callable

from ..op import opdna1_action as action
from .x86_opdna1_base import x86_opdna1

class x86_load(x86_opdna1):
    """
    Base X86 freg and greg loads
    """

    def __init__(self,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        super().__init__(action=action.LOAD, asmwrap=asmwrap, rpref=rpref)
