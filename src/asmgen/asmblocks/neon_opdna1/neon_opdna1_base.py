# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
NEON/ASIMD operations with n data operand and 1 address operand (load/store)
"""

from typing import Callable,Any

from ..aarch64_opdna1.aarch64_opdna1_base import aarch64_opdna1
from ..types.aarch64_types import aarch64_greg,aarch64_freg
from ..op import (
    opdna1,
    opdna1_modifier as mod,
    opdna1_action,
    operation_signature
)
from ...registers import asm_data_type as adt, adt_size

from .signatures import make_neon_opdna1_signatures

class neon_opdna1(opdna1):
    """
    NEON (ASIMD) instruction with n data operands and 1 address operand.
    Inherits from aarch64_opdna1 to automatically handle scalar routing.
    """

    bcast_supported = False

    def __init__(self, action: opdna1_action,
                 asmwrap : Callable[[str],str]):
        self.action = action
        self.asmwrap = asmwrap

        self.scalar_opdna1 = aarch64_opdna1(action=action, asmwrap=asmwrap)

        self.signatures = make_neon_opdna1_signatures(
                bcast_supported=self.bcast_supported)
        self.signatures.extend(self.scalar_opdna1.get_signatures())

    def get_signatures(self) -> list[operation_signature]:
        return self.signatures

    @property
    def inst_base(self):
        """
        Get base root of instruction
        """
        if self.action == opdna1_action.LOAD:
            return "ld"
        if self.action == opdna1_action.STORE:
            return "st"
        raise ValueError(f"Invalid action: {self.action}")


    def diagnose_failure(self, modifiers : set[mod],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):

        unsupported_mods = {
            mod.TINDEX:  (ValueError, "NEON has no ld/st with 2D tile offset indices"),
            mod.VINDEX:  (ValueError, "NEON has no ld/st with 1D vector offset indices"),
            mod.GLANE:   (ValueError, "NEON has no GP-reg lane ld/st"),
            mod.TOFFSET: (ValueError, "NEON has no ld/st with 2D tile offsets"),
            mod.ISTRIDE: (ValueError, "NEON has no ld/st with immediate strides"),
            mod.GSTRIDE: (ValueError, "NEON has no ld/st with GP-reg strides"),
            mod.MASK:    (ValueError, "NEON has no masked ld/st"),
            mod.ROW:     (ValueError, "NEON has no row selection ld/st"),
            mod.COL:     (ValueError, "NEON has no column selection ld/st"),
            mod.NT:      (NotImplementedError, "Non-temporals for NEON not yet implemented"),
        }

        for m, (exc_type, msg) in unsupported_mods.items():
            if m in modifiers:
                raise exc_type(msg)


        if {mod.VOFFSET, mod.IOFFSET} & modifiers:
            if {mod.STRUCT, mod.ILANE, mod.BCAST} & modifiers:
                raise ValueError("VOFFSET/IOFFSET cannot be combined with STRUCT, ILANE, or BCAST")

        if mod.BCAST in modifiers:
            if self.action != opdna1_action.LOAD:
                raise ValueError("BCAST modifier is only valid for LOAD operations")
            if mod.ILANE in modifiers:
                raise ValueError("BCAST cannot be combined with ILANE")

        if mod.POSTINC in modifiers:
            has_iinc = "iinc" in kwargs
            has_increg = "increg" in kwargs

            if has_iinc and has_increg:
                raise ValueError("iinc, increg are mutually exclusive")
            if not has_iinc and not has_increg:
                raise ValueError("Missing one of these parameters: iinc, increg")

        if mod.STRUCT in modifiers and "nstructs" not in kwargs:
            raise ValueError("STRUCT modifier requires 'nstructs' parameter")

        if mod.ILANE in modifiers and "lane" not in kwargs:
            raise ValueError("ILANE modifier requires 'lane' parameter")


    def get_arrangement(self, dt: adt) -> str:
        """
        NEON suffixes for full simd registers
        """
        size = adt_size(dt)
        if size == 1:
            return ".16b"
        if size == 2:
            return ".8h"
        if size == 4:
            return ".4s"
        if size == 8:
            return ".2d"
        raise ValueError(f"Unsupported NEON vector size: {size}")

    def get_element_suffix(self, dt: adt) -> str:
        """
        NEON suffixes for lanes
        """
        size = adt_size(dt)
        if size == 1:
            return ".b"
        if size == 2:
            return ".h"
        if size == 4:
            return ".s"
        if size == 8:
            return ".d"
        raise ValueError(f"Unsupported NEON lane size: {size}")

    def get_addressing(self, areg: aarch64_greg,
                       modifiers: set[mod], **kwargs) -> str:
        """
        Constructs the addressing string

        :param areg: address register
        :param modifiers: operation modifiers
        
        :return: string containing the addressing
        """

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

        addressing = self.get_addressing(agreg, modifiers, **kwargs)

        if mod.VOFFSET in modifiers or mod.IOFFSET in modifiers:
            inst = "ldr" if self.action == opdna1_action.LOAD else "str"
            return self.asmwrap(f"{inst} q{dregs[0].idx}, {addressing}")

        nstructs = kwargs.get("nstructs",1)


        inst = f"{self.inst_base}{nstructs}r" if mod.BCAST in modifiers \
                else f"{self.inst_base}{nstructs}"

        if mod.ILANE in modifiers:
            lane = kwargs.get("lane")
            arrangement = self.get_element_suffix(a_dt)
            reg_list_str = ", ".join([f"{r}{arrangement}" for r in dregs])
            dreg_str = f"{{{reg_list_str}}}[{lane}]"
        else:
            arrangement = self.get_arrangement(a_dt)
            reg_list_str = ", ".join([f"{r}{arrangement}" for r in dregs])
            dreg_str = f"{{{reg_list_str}}}"

        return self.asmwrap(f"{inst} {dreg_str}, {addressing}")
