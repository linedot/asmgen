# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
NEON/ASIMD fma instruction
"""


from ...registers import (
    asm_data_type as adt,
    data_reg,
)
from .neon_opd3_base import neon_opd3_base
from ..types.neon_types import neon_vreg

from ..operations import opd3_modifier as mod

class neon_fma(neon_opd3_base):
    """
    NEON/ASIMD implementation of fma
    """

    inst_base = "ml"
    def check_modifiers(self, modifiers : set[mod]):
        super().check_modifiers(modifiers=modifiers)
        if mod.MASK in modifiers:
            raise ValueError("NEON fma has no masked form")

    def check_valid_registers(self, dregs : list[data_reg]) -> bool:
        if not all(isinstance(d, neon_vreg) for d in dregs):
            raise ValueError("All dregs of a NEON opd3 must be neon_vreg")

    def supported_dts(self) -> list[dict[str,adt]]:
        return [
            {'adreg':adt.FP64, 'bdreg':adt.FP64, 'cdreg':adt.FP64},
            {'adreg':adt.FP32, 'bdreg':adt.FP32, 'cdreg':adt.FP32},
            {'adreg':adt.FP16, 'bdreg':adt.FP16, 'cdreg':adt.FP16},

            {'adreg':adt.FP16, 'bdreg':adt.FP16, 'cdreg':adt.FP32},

            {'adreg':adt.FP8E5M2, 'bdreg':adt.FP8E5M2, 'cdreg':adt.FP32},
            {'adreg':adt.FP8E5M2, 'bdreg':adt.FP8E5M2, 'cdreg':adt.FP16},
            {'adreg':adt.FP8E4M3, 'bdreg':adt.FP8E4M3, 'cdreg':adt.FP32},
            {'adreg':adt.FP8E4M3, 'bdreg':adt.FP8E4M3, 'cdreg':adt.FP16},

            {'adreg':adt.SINT16, 'bdreg':adt.SINT16, 'cdreg':adt.SINT64},
            {'adreg':adt.SINT16, 'bdreg':adt.SINT16, 'cdreg':adt.SINT32},
            {'adreg':adt.SINT8, 'bdreg':adt.SINT8, 'cdreg':adt.SINT32},

            {'adreg':adt.UINT16, 'bdreg':adt.UINT16, 'cdreg':adt.UINT64},
            {'adreg':adt.UINT16, 'bdreg':adt.UINT16, 'cdreg':adt.UINT32},
            {'adreg':adt.UINT8, 'bdreg':adt.UINT8, 'cdreg':adt.UINT32},
        ]
