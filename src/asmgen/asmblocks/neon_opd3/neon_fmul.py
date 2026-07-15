# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
NEON/ASIMD fmul instruction
"""

from ...registers import (
    asm_data_type as adt,
    adt_size,
    data_reg
)
from ..operations import opd3_modifier as mod
from .neon_opd3_base import neon_opd3_base
from ..types.neon_types import neon_vreg

class neon_fmul(neon_opd3_base):
    """
    NEON/ASIMD implementation of fmul
    """

    inst_base = "mul"

    def check_modifiers(self, modifiers : set[mod]):
        super().check_modifiers(modifiers=modifiers)
        if mod.NP in modifiers:
            raise ValueError("NEON mul has no NP-form")
        if mod.MASK in modifiers:
            raise ValueError("NEON mul has no masked form")

    def check_valid_registers(self, dregs : list[data_reg]) -> bool:
        if not all(isinstance(d, neon_vreg) for d in dregs):
            raise ValueError("All dregs of a NEON opd3 must be neon_vreg")

    def check_dts(self, dts : dict[str,adt]):
        super().check_dts(dts)

        if dts['cdreg'] in [adt.UINT64, adt.UINT32, adt.UINT16]:
            if adt_size(dts['adreg']) == adt_size(dts['cdreg']):
                raise ValueError("only widening variants exist for unsigned integer types")


    def supported_dts(self) -> list[dict[str,adt]]:
        return [
            {'adreg':adt.FP64, 'bdreg':adt.FP64, 'cdreg':adt.FP64},
            {'adreg':adt.FP32, 'bdreg':adt.FP32, 'cdreg':adt.FP32},
            {'adreg':adt.FP16, 'bdreg':adt.FP16, 'cdreg':adt.FP16},

            {'adreg':adt.SINT32, 'bdreg':adt.SINT32, 'cdreg':adt.SINT32},
            {'adreg':adt.SINT16, 'bdreg':adt.SINT16, 'cdreg':adt.SINT16},
            {'adreg':adt.SINT8, 'bdreg':adt.SINT8, 'cdreg':adt.SINT8},

            {'adreg':adt.SINT16, 'bdreg':adt.SINT16, 'cdreg':adt.SINT32},
            {'adreg':adt.SINT8, 'bdreg':adt.SINT8, 'cdreg':adt.SINT16},

            {'adreg':adt.UINT16, 'bdreg':adt.UINT16, 'cdreg':adt.UINT32},
            {'adreg':adt.UINT8, 'bdreg':adt.UINT8, 'cdreg':adt.UINT16},

            {'adreg':adt.SINT16, 'bdreg':adt.UINT16, 'cdreg':adt.SINT32},
            {'adreg':adt.SINT8, 'bdreg':adt.UINT8, 'cdreg':adt.SINT16},

            {'adreg':adt.UINT16, 'bdreg':adt.SINT16, 'cdreg':adt.SINT32},
            {'adreg':adt.UINT8, 'bdreg':adt.SINT8, 'cdreg':adt.SINT16},

        ]
