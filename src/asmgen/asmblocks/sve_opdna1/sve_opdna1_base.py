# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
SVE opdna1 operations
"""
import math
from typing import Callable,Any

from ..aarch64_opdna1.aarch64_opdna1_base import aarch64_opdna1
from ..types.aarch64_types import aarch64_greg, aarch64_freg
from ..types.sve_types import sve_preg
from ..op import (
    opdna1,
    opdna1_modifier as mod,
    opdna1_action,
    operation_signature
)

from ...registers import (
    asm_data_type as adt,
    asm_index_type as ait,
    adt_size,
    ait_size,
)

from .signatures import make_sve_opdna1_signatures

class sve_opdna1(opdna1):
    """
    AArch64 SVE instruction with n data operands and 1 address operand.
    Inherits from opdna1 and composes aarch64_opdna1 for scalar routing.
    """

    bcast_supported : bool = False

    def __init__(self, action: opdna1_action, asmwrap: Callable[[str],str]):
        self.action = action
        self.asmwrap = asmwrap
        self.scalar_opdna1 = aarch64_opdna1(action=action, asmwrap=asmwrap)

        self.signatures = make_sve_opdna1_signatures(bcast_supported=self.bcast_supported)
        self.signatures.extend(self.scalar_opdna1.get_signatures())

    def get_signatures(self) -> list[operation_signature]:
        return self.signatures

    @property
    def inst_base(self):
        """
        Get instruction base string
        """
        return "ld" if self.action == opdna1_action.LOAD else "st"


    def diagnose_failure(self, modifiers : set[mod],
                         kwargs : dict[str,Any],
                         dts : dict[str, adt]) -> list[operation_signature]:

        unsupported_mods = {
            mod.TINDEX:  (ValueError, "SVE has no ld/st with 2D tile offset indices"),
            mod.GLANE:   (ValueError, "SVE has no GP-reg lane ld/st"),
            mod.TOFFSET: (ValueError, "SVE has no ld/st with 2D tile offsets"),
            mod.ILANE:   (ValueError, "SVE has no immediate lane ld/st"),
            mod.ISTRIDE: (ValueError, "SVE has no ld/st with immediate strides"),
            mod.GSTRIDE: (ValueError, "SVE has no ld/st with GP-reg strides"),
            mod.ROW:     (ValueError, "SVE has no row selection ld/st"),
            mod.COL:     (ValueError, "SVE has no column selection ld/st"),
            mod.NT:      (NotImplementedError, "Non-temporals for SVE not yet implemented"),
        }

        for m, (exc_type, msg) in unsupported_mods.items():
            if m in modifiers:
                raise exc_type(msg)

        if mod.BCAST in modifiers and self.action != opdna1_action.LOAD:
            raise ValueError("BCAST modifier is only valid for LOAD operations")

        if mod.VINDEX in modifiers and \
          (mod.VOFFSET in modifiers or mod.IOFFSET in modifiers):
            raise ValueError("VINDEX cannot be combined with IOFFSET/VOFFSET")

        if mod.STRUCT in modifiers and "nstructs" not in kwargs:
            raise ValueError("STRUCT modifier requires 'nstructs' parameter")
        if mod.IOFFSET in modifiers and "ioffset" not in kwargs:
            raise ValueError("IOFFSET modifier requires 'ioffset' parameter")
        if mod.VOFFSET in modifiers and "voffset" not in kwargs:
            raise ValueError("VOFFSET modifier requires 'voffset' parameter")
        if mod.GOFFSET in modifiers and "offreg" not in kwargs:
            raise ValueError("GOFFSET modifier requires 'offreg' parameter")
        if mod.VINDEX in modifiers and "it" not in kwargs:
            raise ValueError("VINDEX modifier requires 'it' parameter")
        if mod.VINDEX in modifiers and "vidxreg" not in kwargs:
            raise ValueError("VINDEX modifier requires 'vidxreg' parameter")



    def get_mem_suffix(self, dt: adt) -> str:
        """
        Instruction data size suffix (e.g., ld1w for 32-bit words)
        """
        size = adt_size(dt)
        if size == 1:
            return "b"
        if size == 2:
            return "h"
        if size == 4:
            return "w"
        if size == 8:
            return "d"
        raise ValueError(f"Unsupported SVE memory size: {size}")

    def get_element_suffix(self, dt: adt) -> str:
        """
        Register data size suffix (e.g., z0.s for 32-bit singles)
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
        raise ValueError(f"Unsupported SVE element size: {size}")

    def get_index_suffix(self, it: ait) -> str:
        """
        Register index size suffix (.d or .s)
        """
        size = ait_size(it)
        if size == 4:
            return ".s"
        if size == 8:
            return ".d"
        raise ValueError(f"Unsupported SVE index element size: {size}")

    def get_addressing(self, areg: aarch64_greg,
                       modifiers: set[mod], dt: adt, **kwargs) -> str:
        """
        Generate addressing string
        :param areg: GP reg containing base address
        :param modifiers: operation modifiers
        :param dt: data type to use
        """
        if not isinstance(areg, aarch64_greg):
            raise ValueError(f"{areg} is not an aarch64_greg")

        if mod.VOFFSET in modifiers:
            return f"[{areg}, #{kwargs['voffset']}, MUL VL]"

        if mod.IOFFSET in modifiers:
            return f"[{areg}, #{kwargs['ioffset']}]"

        if mod.GOFFSET in modifiers:
            offreg = kwargs["offreg"]
            size = adt_size(dt)
            shift = ""
            if size > 1:
                lsl = int(math.log2(size))
                shift = f", lsl #{lsl}"
            return f"[{areg}, {offreg}{shift}]"

        if mod.VINDEX in modifiers:
            vidxreg = kwargs["vidxreg"]
            idx_esuf = self.get_index_suffix(kwargs["it"])
            size = adt_size(dt)

            # Scatter/Gather
            if size == 4:
                return f"[{areg}, {vidxreg}{idx_esuf}, sxtw #2]"
            return f"[{areg}, {vidxreg}{idx_esuf}]"

        return f"[{areg}]"

    def implementation(self, *, dregs: list, agreg: aarch64_greg, a_dt: adt,
                       modifiers: set[mod], **kwargs) -> str:

        if not dregs:
            raise ValueError("No dregs provided")

        # Forward scalars to AArch64 base
        if isinstance(dregs[0], (aarch64_greg, aarch64_freg)):
            return self.scalar_opdna1(dregs=dregs, areg=agreg,
                                      dt=a_dt, modifiers=modifiers, **kwargs)

        # 1. Resolve Suffixes
        msuf = self.get_mem_suffix(a_dt)
        esuf = self.get_element_suffix(a_dt)

        # 2. Build Base Instruction (e.g. ld1w, ld2d, ld1rw)
        nstructs = kwargs.get("nstructs", 1)
        if mod.BCAST in modifiers:
            inst = f"{self.inst_base}{nstructs}r{msuf}"
        else:
            inst = f"{self.inst_base}{nstructs}{msuf}"

        # 3. Resolve Predicate (Default to p0 if not passed)
        preg = kwargs.get("amreg", sve_preg(0))
        preg_str = f"{preg}/z" if self.action == opdna1_action.LOAD else f"{preg}"

        # 4. Resolve Registers and Addressing
        dregs_str = ", ".join([f"{r}{esuf}" for r in dregs])
        addressing = self.get_addressing(agreg, modifiers, a_dt, **kwargs)

        return self.asmwrap(f"{inst} {{{dregs_str}}}, {preg_str}, {addressing}")
