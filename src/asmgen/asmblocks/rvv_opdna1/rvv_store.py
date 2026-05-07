# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

"""
RVV 1.0 and 0.7.1 store instructions
"""

from .rvv_opdna1_base import rvv_opdna1
class rvv_store(rvv_opdna1):
    """
    RVV vectore stores
    """

    def __init__(self, lmul_getter :Callable[[],int]):
        super().__init__(inst_base="vs", lmul_getter=lmul_getter)
