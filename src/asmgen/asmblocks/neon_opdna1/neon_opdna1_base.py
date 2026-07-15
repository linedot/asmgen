# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
from ..aarch64_opdna1.aarch64_opdna1_base import aarch64_opdna1
from ..types.neon_types import neon_vreg
from ..types.aarch64_types import aarch64_greg,aarch64_freg
from ..operations import (
    opdna1,
    opdna1_modifier as mod,
    opdna1_action,
    operand_restriction
)
from ...registers import asm_data_type as adt, adt_size

from typing import Callable

class neon_opdna1(opdna1):
    """
    NEON (ASIMD) instruction with n data operands and 1 address operand.
    Inherits from aarch64_opdna1 to automatically handle scalar routing.
    """

    def __init__(self, action: opdna1_action,
                 asmwrap : Callable[[str],str]):
        self.action = action
        self.asmwrap = asmwrap

        self.scalar_opdna1 = aarch64_opdna1(action=action, asmwrap=asmwrap)

    @property
    def inst_base(self):
        if self.action == opdna1_action.LOAD:
            return "ld"
        elif self.action == opdna1_action.STORE:
            return "st"
        else:
            raise ValueError(f"Invalid action: {self.action}")

    def supported_dts(self) -> list[dict[str,adt]]:
        return [
            {'adreg': adt.FP64, 'bdreg': adt.FP64, 'cdreg': adt.FP64, 'ddreg': adt.FP64},
            {'adreg': adt.FP32, 'bdreg': adt.FP32, 'cdreg': adt.FP32, 'ddreg': adt.FP32},
            {'adreg': adt.FP16, 'bdreg': adt.FP16, 'cdreg': adt.FP16, 'ddreg': adt.FP16},
            {'adreg': adt.FP8E4M3, 'bdreg': adt.FP8E4M3, 'cdreg': adt.FP8E4M3, 'ddreg': adt.FP8E4M3},
            {'adreg': adt.FP8E5M2, 'bdreg': adt.FP8E5M2, 'cdreg': adt.FP8E5M2, 'ddreg': adt.FP8E5M2},
            {'adreg': adt.SINT64, 'bdreg': adt.SINT64, 'cdreg': adt.SINT64, 'ddreg': adt.SINT64},
            {'adreg': adt.SINT32, 'bdreg': adt.SINT32, 'cdreg': adt.SINT32, 'ddreg': adt.SINT32},
            {'adreg': adt.SINT16, 'bdreg': adt.SINT16, 'cdreg': adt.SINT16, 'ddreg': adt.SINT16},
            {'adreg': adt.SINT8, 'bdreg': adt.SINT8, 'cdreg': adt.SINT8, 'ddreg': adt.SINT8},
            {'adreg': adt.UINT64, 'bdreg': adt.UINT64, 'cdreg': adt.UINT64, 'ddreg': adt.UINT64},
            {'adreg': adt.UINT32, 'bdreg': adt.UINT32, 'cdreg': adt.UINT32, 'ddreg': adt.UINT32},
            {'adreg': adt.UINT16, 'bdreg': adt.UINT16, 'cdreg': adt.UINT16, 'ddreg': adt.UINT16},
            {'adreg': adt.UINT8, 'bdreg': adt.UINT8, 'cdreg': adt.UINT8, 'ddreg': adt.UINT8}
        ]

    def check_modifiers(self, modifiers: set[mod]):

        if mod.TINDEX in modifiers:
            raise ValueError(
                    "NEON has no ld/st with 2D tile offset indices")
        if mod.VINDEX in modifiers:
            raise ValueError(
                    "NEON has no ld/st with 1D vector offset indices")
        if mod.GLANE in modifiers:
            raise ValueError("NEON has no GP-reg lane ld/st")
        if mod.TOFFSET in modifiers:
            raise ValueError("NEON has no ld/st with 2D tile offsets")
        if mod.ISTRIDE in modifiers:
            raise ValueError(
                    "NEON has no ld/st with immediate strides")
        if mod.GSTRIDE in modifiers:
            raise ValueError("NEON has no ld/st with GP-reg strides")
        if mod.MASK in modifiers:
            raise ValueError("NEON has no masked ld/st")
        if mod.ROW in modifiers:
            raise ValueError("NEON has no row selection ld/st")
        if mod.COL in modifiers:
            raise ValueError("NEON has no column selection ld/st")
        if mod.NT in modifiers:
            raise NotImplementedError("Non-temporals for NEON not yet implemented")
                
        # Modifier/Kwarg Compatibility Checks
        if mod.VOFFSET in modifiers or mod.IOFFSET in modifiers:
            if any(m in modifiers for m in [mod.STRUCT, mod.ILANE, mod.BCAST]):
                raise ValueError("VOFFSET/IOFFSET cannot be combined with STRUCT, ILANE, or BCAST")
        
        if mod.BCAST in modifiers:
            if self.action != opdna1_action.LOAD:
                raise ValueError("BCAST modifier is only valid for LOAD operations")
            if mod.ILANE in modifiers:
                raise ValueError("BCAST cannot be combined with ILANE")

    def get_operand_restrictions(self, oprnd : str) -> set[operand_restriction]:
        rstrs = {
            'bdreg' : {operand_restriction.IDXOTHERPLUSN},
            'cdreg' : {operand_restriction.IDXOTHERPLUSN},
            'ddreg' : {operand_restriction.IDXOTHERPLUSN},
        }

        if oprnd in rstrs:
            return rstrs[oprnd]
        return set()

    def get_operand_restriction_value(self, oprnd : str,
                                      modifiers : set[mod],
                                      rstr : operand_restriction) \
      -> int|set[int]|tuple[str,int]:

        if oprnd in {'bdreg', 'cdreg', 'ddreg'} and \
          rstr == operand_restriction.IDXOTHERPLUSN:
            return (chr(ord(oprnd[0])-1)+'dreg', 1)

        raise ValueError("No restriction {rstr} on operand {op} for NEON opd3")

    def get_required_params(self, modifiers: set[mod]) -> list[str]:

        required_extra_params = []

        if mod.IOFFSET in modifiers:
            required_extra_params.append({"ioffset"})

        if mod.ILANE in modifiers:
            required_extra_params.append({"lane"})

        if mod.GOFFSET in modifiers:
            required_extra_params.append({"offreg"})

        if mod.STRUCT in modifiers:
            required_extra_params.append({"nstructs"})

        if mod.POSTINC in modifiers:
            required_extra_params.append({"iinc","increg"})

        return required_extra_params


    def get_arrangement(self, dt: adt) -> str:
        size = adt_size(dt)
        if size == 1: return ".16b"
        if size == 2: return ".8h"
        if size == 4: return ".4s"
        if size == 8: return ".2d"
        raise ValueError(f"Unsupported NEON vector size: {size}")

    def get_element_suffix(self, dt: adt) -> str:
        size = adt_size(dt)
        if size == 1: return ".b"
        if size == 2: return ".h"
        if size == 4: return ".s"
        if size == 8: return ".d"
        raise ValueError(f"Unsupported NEON lane size: {size}")

    def get_addressing(self, areg: aarch64_greg, modifiers: set[mod], **kwargs) -> str:
        if not isinstance(areg, aarch64_greg):
            raise ValueError(f"{areg} is not an aarch64_greg")

        base_addr = f"[{areg}]"

        if mod.VOFFSET in modifiers or mod.IOFFSET in modifiers:
            byte_offset = kwargs.get("ioffset", 0)
            if mod.VOFFSET in modifiers:
                byte_offset = kwargs["voffset"] * 16
            if byte_offset == 0:
                return base_addr
            return f"[{areg}, #{byte_offset}]"

        if mod.POSTINC in modifiers:
            increg = kwargs.get("increg")
            if increg is not None:
                if not isinstance(increg, aarch64_greg):
                    raise ValueError(f"{increg} is not an aarch64_greg")
                return f"{base_addr}, {increg}"
            iinc = kwargs["iinc"]
            return f"{base_addr}, #{iinc}"

        return base_addr

    def implementation(self, *,
                       dregs: list, agreg: aarch64_greg, a_dt: adt,
                       modifiers: set[mod], **kwargs) -> str:
                 
        if not dregs:
            raise ValueError("No dregs provided")

        # If scalar registers are passed, forward to base AArch64
        if isinstance(dregs[0], (aarch64_greg, aarch64_freg)):
            return self.scalar_opdna1(dregs=dregs, areg=agreg, dt=a_dt,
                                      modifiers=modifiers, **kwargs)

        # Vector Logic Enforced Here
        if not all(isinstance(reg, neon_vreg) for reg in dregs):
            raise ValueError("Mixed or invalid register types for NEON vector operation")

        addressing = self.get_addressing(agreg, modifiers, **kwargs)

        # Case 1: LDR / STR for IOFFSET / VOFFSET
        if mod.VOFFSET in modifiers or mod.IOFFSET in modifiers:
            inst = "ldr" if self.action == opdna1_action.LOAD else "str"
            return self.asmwrap(f"{inst} q{dregs[0].idx}, {addressing}")

        # Segmented registers contiguity check
        for i in range(1, len(dregs)):
            if dregs[i].idx != (dregs[i-1].idx + 1) % 32:
                raise ValueError("NEON segmented registers must be contiguous.")

        nstructs = kwargs.get("nstructs",1)
        # Checks not covered by standard parameter tests
        if mod.STRUCT not in modifiers and len(dregs) != 1:
            raise ValueError(
                    "Multiple registers provided but STRUCT modifier is missing")
        if mod.STRUCT in modifiers and len(dregs) != nstructs:
            raise ValueError(
                    f"Number of dregs differs from nstructs ({len(dregs)} != {nstructs})")

        inst = f"{self.inst_base}{nstructs}r" if mod.BCAST in modifiers else f"{self.inst_base}{nstructs}"
        
        if mod.ILANE in modifiers:
            lane = kwargs.get("lane")
            if lane is None: raise ValueError("ILANE requires 'lane' parameter")
            arrangement = self.get_element_suffix(a_dt)
            reg_list_str = ", ".join([f"{r}{arrangement}" for r in dregs])
            dreg_str = f"{{{reg_list_str}}}[{lane}]"
        else:
            arrangement = self.get_arrangement(a_dt)
            reg_list_str = ", ".join([f"{r}{arrangement}" for r in dregs])
            dreg_str = f"{{{reg_list_str}}}"

        return self.asmwrap(f"{inst} {dreg_str}, {addressing}")
