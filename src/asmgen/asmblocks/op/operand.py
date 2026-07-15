# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Operands
"""

from dataclasses import dataclass,field
from enum import Enum, auto

from typing import Any


from ...registers import (
    asm_data_type as adt,
    greg_base,
    freg_base,
    vreg_base,
    treg_base,
    mreg_base
)

from .constraint import operand_constraint

class operand_type(Enum):
    REGISTER = auto()
    IMMEDIATE = auto()


class register_type(Enum):
    GP = auto()
    FP = auto()
    VEC = auto()
    TILE = auto()
    MASK = auto()


def is_register_type(val : Any, rt : register_type) -> bool:
    if register_type.GP   == rt and isinstance(val, greg_base):
        return True       
    if register_type.FP   == rt and isinstance(val, freg_base):
        return True       
    if register_type.VEC  == rt and isinstance(val, vreg_base):
        return True
    if register_type.TILE == rt and isinstance(val, treg_base):
        return True
    if register_type.MASK == rt and isinstance(val, mreg_base):
        return True

    return False

@dataclass
class operand_shape:
    """
    Structural requirement for an operand
    """

    otype : operand_type
    rtype : register_class | None = None
    dt    : adt = None

    value_constraints: list[operand_constraint] = field(default_factory=list)
