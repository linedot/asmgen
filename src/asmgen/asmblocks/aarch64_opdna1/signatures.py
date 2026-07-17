# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Valid signatures for AArch64  opdna1 operations
"""

from ..op import (
    operation_signature as sig,
    operand_shape as osh,
    operand_type as ot,
    register_type as rt,
    opdna1_modifier as mod
)

from ...registers import asm_data_type as adt, adt_is_float

_FLOATS = [adt.FP64, adt.FP32, adt.FP16, adt.FP8E4M3, adt.FP8E5M2]
_INTS = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8,
         adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]

def make_aarch64_opdna1_signatures():
    """
    Generate signatures for AArch64 opdna1 operations
    """
    sigs = []


    for dt in _FLOATS+_INTS:

        a_rt = rt.FP if adt_is_float(dt) else rt.GP

        # (f)ld without offset
        sigs.append(sig(
        modifiers=set(),
        structural_params={},
        operands={
            'adreg': osh(ot.REGISTER, a_rt, dt),
            'agreg': osh(ot.REGISTER, rt.GP, adt.UINT64)
            }
        ))
        # (f)ld with immediate offset
        sigs.append(sig(
        modifiers={mod.IOFFSET},
        structural_params={},
        operands={
            'adreg': osh(ot.REGISTER, a_rt, dt),
            'agreg': osh(ot.REGISTER, rt.GP, adt.UINT64),
            'ioffset': osh(ot.IMMEDIATE, None, adt.UINT64)
            }
        ))
        # (f)ld with gp-reg offset
        sigs.append(sig(
        modifiers={mod.GOFFSET},
        structural_params={},
        operands={
            'adreg': osh(ot.REGISTER, a_rt, dt),
            'agreg': osh(ot.REGISTER, rt.GP, adt.UINT64),
            'offreg': osh(ot.REGISTER, rt.GP, adt.UINT64)
            }
        ))
        # (f)ld with postinc immediate offset
        sigs.append(sig(
        modifiers={mod.POSTINC},
        structural_params={},
        operands={
            'adreg': osh(ot.REGISTER, a_rt, dt),
            'agreg': osh(ot.REGISTER, rt.GP, adt.UINT64),
            'iinc': osh(ot.IMMEDIATE, None, adt.UINT64)
            }
        ))
        # (f)ld with postinc gp-reg offset
        sigs.append(sig(
        modifiers={mod.POSTINC},
        structural_params={},
        operands={
            'adreg': osh(ot.REGISTER, a_rt, dt),
            'agreg': osh(ot.REGISTER, rt.GP, adt.UINT64),
            'increg': osh(ot.REGISTER, rt.GP, adt.UINT64)
            }
        ))

    return sigs
