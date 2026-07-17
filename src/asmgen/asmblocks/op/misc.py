# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Helpers for dealing with operands
"""

def make_ord_prefix(i : int) -> 'str':
    """
    Maps 0-25 onto a-z
    """
    if i > 25 or i < 0:
        raise ValueError("index outside of allowed range [0,25]")

    return chr(ord('a')+i)
