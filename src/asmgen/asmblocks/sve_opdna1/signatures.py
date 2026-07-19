# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Valid signatures for SVE opdna1 operations
"""

from dataclasses import dataclass
from typing import Callable

from ..op import (
    operation_signature as sig,
    operand_shape as osh,
    operand_type as ot,
    register_type as rt,
    opdna1_modifier as mod
)

from ..op.constraint import (
    otherplusnmod_constraint,
)
from ..op.misc import make_ord_prefix as mop

from ..types.sve_types import sve_vreg

from ...registers import (
    asm_data_type as adt,
    asm_index_type as ait,
    adt_size
)

_FLOATS = [adt.FP64, adt.FP32, adt.FP16, adt.BF16, adt.FP8E4M3, adt.FP8E5M2]
_INTS = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8,
         adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]

INDEX_ADT_SIZE_MAP = {
    8 : adt.SINT64,
    4 : adt.SINT32,
}
INDEX_AIT_SIZE_MAP = {
    8 : ait.INT64,
    4 : ait.INT32,
}

@dataclass(kw_only=True)
class sve_struct_constraint(otherplusnmod_constraint):
    """
    Constraint ensuring structured loads/stores use consecutive registers
    """
    what : str = 'index'
    getint : Callable[[sve_vreg],int] = lambda reg : reg.idx
    makeval : Callable[[int],sve_vreg] = lambda idx : sve_vreg(reg_idx=idx)
    offset : int = 1
    modval : int = 32

def make_sve_opdna1_signatures(bcast_supported=False):
    """
    Generate signatures for NEON opdna1 operations

    :param bcast_supported: whether the instruction supports broadcasts (loads only)
    """
    sigs = []

    def add_sig(dt, *, mods, nstructs=1):
        ops = {
            'adreg': osh(ot.REGISTER, rt.VEC, dt),
            'agreg': osh(ot.REGISTER, rt.GP, dt.UINT64),
            'amreg': osh(ot.REGISTER, rt.MASK, dt)
        }

        struct_params={}

        # structured ld/st logic
        if mod.STRUCT in mods:
            struct_params['nstructs'] = nstructs
            for i in range(1, nstructs):
                ops[f"{mop(i)}dreg"] = osh(
                    ot.REGISTER, rt.VEC, dt,
                    value_constraints=[
                        sve_struct_constraint(other=f"{mop(i-1)}dreg")
                    ])

        if mod.VINDEX in mods:
            ops['vidxreg'] = osh(ot.REGISTER, rt.VEC, INDEX_ADT_SIZE_MAP[adt_size(dt)])
            struct_params['it'] = INDEX_AIT_SIZE_MAP[adt_size(dt)]

        if mod.IOFFSET in mods:
            ops['ioffset'] = osh(ot.IMMEDIATE, None, None)
        if mod.GOFFSET in mods:
            ops['offreg'] = osh(ot.REGISTER, rt.GP, adt.SINT64)
        if mod.VOFFSET in mods:
            ops['voffset'] = osh(ot.IMMEDIATE, None, None)

        sigs.append(sig(
            modifiers=mods,
            structural_params=struct_params,
            operands=ops
        ))

    for dt in _FLOATS+_INTS:
        add_sig(dt, mods={mod.MASK})
        if adt_size(dt) >= 4:
            # Gathers always use 32 bit or 64 bit indices,
            # for fp16/bf16/fp8/etc... it means filling the vector with one instruction is
            # not possible. it also places the 16/8 bit values into the low bits of
            # 32bit/64bit lanes. Will need to figure out how to handle this.
            add_sig(dt, mods={mod.MASK, mod.VINDEX})
            add_sig(dt, mods={mod.MASK, mod.VINDEX, mod.IOFFSET})
        add_sig(dt, mods={mod.MASK, mod.VOFFSET})
        add_sig(dt, mods={mod.MASK, mod.GOFFSET})
        if bcast_supported:
            add_sig(dt, mods={mod.MASK, mod.BCAST})
        for nstructs in range(2,5):
            add_sig(dt, mods={mod.MASK, mod.STRUCT},nstructs=nstructs)
            if bcast_supported:
                add_sig(dt, mods={mod.MASK, mod.BCAST,mod.STRUCT},nstructs=nstructs)

    return sigs
