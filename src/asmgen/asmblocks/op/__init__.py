# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Operation/Instruction abstractions and utilities
"""


from .constraint import operand_constraint
from .modifier import operation_modifier
from .signature import operation_signature

from .operation import (
    operation,
)

from .operand import operand_shape, operand_type, register_type

from .opdna1 import (
    opdna1_modifier,
    opdna1_action,
    opdna1
)

from .opd3 import (
    widening_method,
    opd3_modifier,
    opd3,
    dummy_opd3)
