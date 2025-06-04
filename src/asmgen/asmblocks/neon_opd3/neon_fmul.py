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
    adt_triple,
    adt_size,
    adt_is_float,
)
from ..operations import modifier

from .neon_opd3_base import neon_opd3_base

class neon_fmul(neon_opd3_base):
    """
    NEON/ASIMD implementation of fmul
    """

    inst_base = "mul"

    def check_modifiers(self, modifiers : set[modifier]):
        super().check_modifiers(modifiers=modifiers)
        if modifier.NP in modifiers:
            raise ValueError("NEON mul has no NP-form")

    def check_triple(self, a_dt : adt, b_dt : adt, c_dt : adt):
        super().check_triple(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)

        if adt_is_float(c_dt) and (adt_size(c_dt) > adt_size(a_dt)):
            raise ValueError("NEON FMUL has no widening FP version")

        if adt_size(c_dt)//adt_size(a_dt) > 2:
            raise ValueError("NEON FMUL has only 2xways widening")

        if c_dt in [adt.UINT64, adt.UINT32, adt.UINT16]:
            if adt_size(a_dt) == adt_size(c_dt):
                raise ValueError("only widening variants exist for unsigned integer types")


    def supported_triples(self) -> list[adt_triple]:
        return [
            adt_triple(a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64),
            adt_triple(a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32),
            adt_triple(a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16),

            adt_triple(a_dt=adt.SINT32, b_dt=adt.SINT32, c_dt=adt.SINT32),
            adt_triple(a_dt=adt.SINT16, b_dt=adt.SINT16, c_dt=adt.SINT16),
            adt_triple(a_dt=adt.SINT8,  b_dt=adt.SINT8,  c_dt=adt.SINT8),

            adt_triple(a_dt=adt.SINT16, b_dt=adt.SINT16, c_dt=adt.SINT32),
            adt_triple(a_dt=adt.SINT8, b_dt=adt.SINT8, c_dt=adt.SINT16),

            adt_triple(a_dt=adt.UINT16, b_dt=adt.UINT16, c_dt=adt.UINT32),
            adt_triple(a_dt=adt.UINT8, b_dt=adt.UINT8, c_dt=adt.UINT16),

            adt_triple(a_dt=adt.SINT16, b_dt=adt.UINT16, c_dt=adt.SINT32),
            adt_triple(a_dt=adt.SINT8, b_dt=adt.UINT8, c_dt=adt.SINT16),

            adt_triple(a_dt=adt.UINT16, b_dt=adt.SINT16, c_dt=adt.SINT32),
            adt_triple(a_dt=adt.UINT8, b_dt=adt.SINT8, c_dt=adt.SINT16),

        ]
