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
    adt_triple,
)
from .neon_opd3_base import neon_opd3_base

class neon_fma(neon_opd3_base):
    """
    NEON/ASIMD implementation of fma
    """

    inst_base = "ml"

    def supported_triples(self) -> list[adt_triple]:
        return [
            adt_triple(a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64),
            adt_triple(a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32),
            adt_triple(a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16),

            adt_triple(a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP32),

            adt_triple(a_dt=adt.FP8E5M2, b_dt=adt.FP8E5M2, c_dt=adt.FP32),
            adt_triple(a_dt=adt.FP8E5M2, b_dt=adt.FP8E5M2, c_dt=adt.FP16),
            adt_triple(a_dt=adt.FP8E4M3, b_dt=adt.FP8E4M3, c_dt=adt.FP32),
            adt_triple(a_dt=adt.FP8E4M3, b_dt=adt.FP8E4M3, c_dt=adt.FP16),

            adt_triple(a_dt=adt.SINT16, b_dt=adt.SINT16, c_dt=adt.SINT64),
            adt_triple(a_dt=adt.SINT16, b_dt=adt.SINT16, c_dt=adt.SINT32),
            adt_triple(a_dt=adt.SINT8,  b_dt=adt.SINT8,  c_dt=adt.SINT32),

            adt_triple(a_dt=adt.UINT16, b_dt=adt.UINT16, c_dt=adt.UINT64),
            adt_triple(a_dt=adt.UINT16, b_dt=adt.UINT16, c_dt=adt.UINT32),
            adt_triple(a_dt=adt.UINT8,  b_dt=adt.UINT8,  c_dt=adt.UINT32),
        ]
