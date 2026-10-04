# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
SVE store instructions
"""

from typing import Callable

from ..op import opmem_action as action
from .sve_opmem_base import sve_opmem

class sve_store(sve_opmem):
    """
    SVE register stores
    """

    def __init__(self, asmwrap : Callable[[str],str]):
        super().__init__(action=action.STORE, asmwrap=asmwrap)
