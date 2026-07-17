# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Valid signatures for NEON opdna1 operations
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
    minmax_constraint
)
from ..op.misc import make_ord_prefix as mop

from ..types.neon_types import neon_vreg

from ...registers import (
    asm_data_type as adt,
    adt_size
)

_FLOATS = [adt.FP64, adt.FP32, adt.FP16, adt.FP8E4M3, adt.FP8E5M2]
_INTS = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8,
         adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]

@dataclass(kw_only=True)
class neon_struct_constraint(otherplusnmod_constraint):
    """
    Constraint ensuring structured loads/stores use consecutive registers
    """
    what : str = 'index'
    getint : Callable[[neon_vreg],int] = lambda reg : reg.idx
    makeval : Callable[[int],neon_vreg] = lambda idx : neon_vreg(reg_idx=idx)
    offset : int = 1
    modval : int = 32

def make_neon_opdna1_signatures(bcast_supported=False):
    """
    Generate signatures for NEON opdna1 operations

    :param bcast_supported: whether the instruction supports broadcasts (loads only)
    """
    sigs = []

    def add_sig(dt, *, mods, nstructs=1, postinc_reg=False):
        ops = {
            'adreg': osh(ot.REGISTER, rt.VEC, dt),
            'agreg': osh(ot.REGISTER, rt.GP, dt.UINT64),
        }

        struct_params={}

        # structured ld/st logic
        if mod.STRUCT in mods:
            struct_params['nstructs'] = nstructs
            for i in range(1, nstructs):
                ops[f"{mop(i)}dreg"] = osh(
                    ot.REGISTER, rt.VEC, dt,
                    value_constraints=[
                        neon_struct_constraint(other=f"{mop(i-1)}dreg")
                    ])

        if mod.IOFFSET in mods:
            ops['ioffset'] = osh(ot.IMMEDIATE, None, None)
        if mod.VOFFSET in mods:
            ops['voffset'] = osh(ot.IMMEDIATE, None, None)
        if mod.ILANE in mods:
            max_lane = (16 // adt_size(dt))-1
            ops['lane'] = osh(
                ot.IMMEDIATE, None, None,
                value_constraints=[minmax_constraint(minval=0,maxval=max_lane)])

        if mod.POSTINC in mods:
            if postinc_reg:
                ops['increg'] = osh(ot.REGISTER, rt.GP, adt.UINT64)
            else:
                ops['iinc'] = osh(ot.IMMEDIATE, None, None)

        sigs.append(sig(
            modifiers=mods,
            structural_params=struct_params,
            operands=ops
        ))

    for dt in _FLOATS+_INTS:
        add_sig(dt, mods=set())
        add_sig(dt, mods={mod.IOFFSET})
        add_sig(dt, mods={mod.VOFFSET})
        add_sig(dt, mods={mod.ILANE})
        add_sig(dt, mods={mod.ILANE,mod.VOFFSET})
        add_sig(dt, mods={mod.POSTINC},postinc_reg=False)
        add_sig(dt, mods={mod.POSTINC},postinc_reg=True)
        if bcast_supported:
            add_sig(dt, mods={mod.BCAST})
            # No BCAST with offsets, but POSTINC is allowed
            add_sig(dt, mods={mod.BCAST,mod.POSTINC},postinc_reg=False)
            add_sig(dt, mods={mod.BCAST,mod.POSTINC},postinc_reg=True)
        for nstructs in range(2,5):
            add_sig(dt, mods={mod.STRUCT},nstructs=nstructs)
            add_sig(dt, mods={mod.STRUCT,mod.POSTINC},
                    nstructs=nstructs, postinc_reg=False)
            add_sig(dt, mods={mod.STRUCT,mod.POSTINC},
                    nstructs=nstructs, postinc_reg=True)
            if bcast_supported:
                add_sig(dt, mods={mod.BCAST,mod.STRUCT},nstructs=nstructs)
                add_sig(dt, mods={mod.BCAST,mod.STRUCT,mod.POSTINC},
                        nstructs=nstructs, postinc_reg=False)
                add_sig(dt, mods={mod.BCAST,mod.STRUCT,mod.POSTINC},
                        nstructs=nstructs, postinc_reg=True)

    return sigs
