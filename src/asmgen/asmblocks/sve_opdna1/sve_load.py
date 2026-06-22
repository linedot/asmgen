# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
SVE load instructions
"""

from ..operations import opdna1_action as action
from .sve_opdna1_base import sve_opdna1

from typing import Callable

class sve_load(sve_opdna1):
    """
    SVE register loads
    """

    def __init__(self, asmwrap : Callable[[str],str]):
        super().__init__(action=action.LOAD, asmwrap=asmwrap)
