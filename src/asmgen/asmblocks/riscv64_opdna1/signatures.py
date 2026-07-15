# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Valid signatures for RISCV64 +D/F opdna1 operations
"""

from ..op import (
    operation_signature as sig,
    operand_shape as osh,
    operand_type as ot,
    register_type as rt,
    opdna1_modifier as mod
)

from ...registers import asm_data_type as adt


def make_riscv64_opdna1_signatures():
    sigs = []

    floats = [adt.FP64, adt.FP32, adt.FP16, adt.FP8E4M3, adt.FP8E5M2]
    ints = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8,
            adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]
    
    for dt in floats:
        # normal fld without offset
        sigs.append(sig(
        modifiers={},
        structural_params={},
        operands={
            'adreg': osh(ot.REGISTER, rt.FP, dt),
            'agreg': osh(ot.REGISTER, rt.GP, adt.UINT64)
            }
        ))
        # normal fld with immediate offset
        sigs.append(sig(
        modifiers={mod.IOFFSET},
        structural_params={},
        operands={
            'adreg': osh(ot.REGISTER, rt.FP, dt),
            'agreg': osh(ot.REGISTER, rt.GP, adt.UINT64),
            'ioffset': osh(ot.IMMEDIATE, None, adt.UINT64)
            }
        ))

    # Same for integers
    for dt in ints:
        sigs.append(sig(
        modifiers={},
        structural_params={},
        operands={
            'adreg': osh(ot.REGISTER, rt.GP, dt),
            'agreg': osh(ot.REGISTER, rt.GP, adt.UINT64)
            }
        ))
        sigs.append(sig(
        modifiers={mod.IOFFSET},
        structural_params={},
        operands={
            'adreg': osh(ot.REGISTER, rt.GP, dt),
            'agreg': osh(ot.REGISTER, rt.GP, adt.UINT64),
            'ioffset': osh(ot.IMMEDIATE, None, adt.UINT64)
            }
        ))

    return sigs


