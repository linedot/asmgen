# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RVV 1.0 and 0.7.1 load instructions
"""

from ..operations import opdna1_action as action
from .rvv_opdna1_base import rvv_opdna1

from typing import Callable

class rvv_load(rvv_opdna1):
    """
    RVV vector loads
    """

    def __init__(self,
                 asmwrap : Callable[[str],str],
                 lmul_getter :Callable[[],int]):
        super().__init__(action=action.LOAD,
                         asmwrap=asmwrap,
                         lmul_getter=lmul_getter)
