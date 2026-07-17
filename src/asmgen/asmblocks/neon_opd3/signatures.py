
# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Valid signatures for NEON opd3 operations
"""
from ..op import (
    operation_signature as sig,
    operand_shape as osh,
    operand_type as ot,
    register_type as rt,
    opd3_modifier as mod
)

from ..op.constraint import minmax_constraint
from ..op.opd3 import widening_method as wm
from ...registers import asm_data_type as adt, adt_size
from ..types.neon_types import neon_vreg

_FLOATS = [adt.FP64, adt.FP32, adt.FP16, adt.FP8E4M3, adt.FP8E5M2]
_SIGNED_INTS = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8]
_UNSIGNED_INTS = [adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]

_WIDENING_2X_MAP = {
    adt.FP8E4M3: adt.FP16,   adt.FP8E5M2: adt.FP16,
    adt.FP16: adt.FP32,
    adt.UINT8: adt.UINT16,   adt.SINT8: adt.SINT16,
    adt.UINT16: adt.UINT32,  adt.SINT16: adt.SINT32,
}
_WIDENING_4X_MAP = {
    adt.FP8E4M3: adt.FP32, adt.FP8E5M2: adt.FP32,
    adt.UINT8: adt.UINT32, adt.SINT8: adt.SINT32
}

_MIXED_INTS = [
    (adt.UINT8, adt.SINT8, adt.SINT16), (adt.SINT8, adt.UINT8, adt.SINT16),
    (adt.UINT16, adt.SINT16, adt.SINT32), (adt.SINT16, adt.UINT16, adt.SINT32),
]

# Readable enough, no need for subfunctions
# pylint: disable-next=too-many-branches
def make_neon_opd3_signatures(supports_np: bool) -> list[sig]:
    """
    Generate signatures for NEON opd3 operations
    """
    sigs = []

    base_mods = [set(), {mod.IDX}]
    if supports_np:
        base_mods.extend([{mod.NP}, {mod.NP, mod.IDX}])

    def add_sig(a_dt, b_dt, c_dt, *, mods, is_widening=False):
        struct_params = {'widening_method': wm.SPLIT_INSTRUCTIONS} if is_widening else {}

        ops = {
            'adreg': osh(ot.REGISTER, rt.VEC, a_dt),
            'bdreg': osh(ot.REGISTER, rt.VEC, b_dt),
            'cdreg': osh(ot.REGISTER, rt.VEC, c_dt)
        }

        if mod.IDX in mods:
            max_idx = (16 // adt_size(b_dt)) - 1
            ops['idx'] = osh(
                ot.IMMEDIATE, None, None,
                value_constraints=[minmax_constraint(minval=0, maxval=max_idx)]
            )
            # with 16bit indexed fma, b has to be v0-v15
            if adt_size(b_dt) <= 2:
                ops['bdreg'].value_constraints.append(
                        minmax_constraint(
                            what='index',
                            getint=lambda reg : reg.idx,
                            makeval=lambda idx : neon_vreg(reg_idx=idx),
                            minval=0, maxval=15
                        )
                )
        if mod.PART in mods:
            max_part = (adt_size(c_dt) // adt_size(a_dt)) - 1
            ops['part'] = osh(
                ot.IMMEDIATE, None, None,
                value_constraints=[minmax_constraint(minval=0, maxval=max_part)]
            )

        sigs.append(sig(
            modifiers=mods,
            structural_params=struct_params,
            operands=ops
        ))

    for dt in _FLOATS:
        for m in base_mods:
            add_sig(dt, dt, dt, mods=m)
            if dt in _WIDENING_2X_MAP:
                add_sig(dt, dt, _WIDENING_2X_MAP[dt], mods=m | {mod.PART}, is_widening=True)
            if dt in _WIDENING_4X_MAP:
                add_sig(dt, dt, _WIDENING_4X_MAP[dt], mods=m | {mod.PART}, is_widening=True)

    for dt in _SIGNED_INTS:
        for m in base_mods:
            add_sig(dt, dt, dt, mods=m)
            if dt in _WIDENING_2X_MAP:
                add_sig(dt, dt, _WIDENING_2X_MAP[dt], mods=m | {mod.PART}, is_widening=True)
            if dt in _WIDENING_4X_MAP:
                add_sig(dt, dt, _WIDENING_4X_MAP[dt], mods=m | {mod.PART}, is_widening=True)

    for dt in _UNSIGNED_INTS:
        for m in base_mods:
            # Widening only for unsigned ints
            if dt in _WIDENING_2X_MAP:
                add_sig(dt, dt, _WIDENING_2X_MAP[dt], mods=m | {mod.PART}, is_widening=True)
            if dt in _WIDENING_4X_MAP:
                add_sig(dt, dt, _WIDENING_4X_MAP[dt], mods=m | {mod.PART}, is_widening=True)

    for a_dt, b_dt, c_dt in _MIXED_INTS:
        for m in [set(), {mod.IDX}]:
            add_sig(a_dt, b_dt, c_dt, mods=m | {mod.PART}, is_widening=True)

    return sigs
