# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
import math
from typing import Callable

from ..aarch64_opdna1.aarch64_opdna1_base import aarch64_opdna1
from ..types.aarch64_types import aarch64_greg, aarch64_freg
from ..types.sve_types import sve_vreg, sve_preg
from ..operations import opdna1_modifier as mod, opdna1_action, opdna1
from ...registers import asm_data_type as adt, adt_size, data_reg

class sve_opdna1(opdna1):
    """
    AArch64 SVE instruction with n data operands and 1 address operand.
    Inherits from opdna1 and composes aarch64_opdna1 for scalar routing.
    """

    def __init__(self, action: opdna1_action, asmwrap: Callable[[str],str]):
        self.action = action
        self.asmwrap = asmwrap
        self.scalar_opdna1 = aarch64_opdna1(action=action, asmwrap=asmwrap)

    @property
    def inst_base(self):
        return "ld" if self.action == opdna1_action.LOAD else "st"

    def supported_dts(self) -> list[adt]:
        return [
            adt.FP64, adt.FP32, adt.FP16, adt.BF16, adt.FP8E4M3, adt.FP8E5M2,
            adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8,
            adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8
        ]

    def check_modifiers(self, modifiers: set[mod]):
        if mod.TINDEX in modifiers:
            raise ValueError("SVE has no ld/st with 2D tile offset indices")
        if mod.GLANE in modifiers:
            raise ValueError("SVE has no GP-reg lane ld/st")
        if mod.ILANE in modifiers:
            raise ValueError("SVE has no immediate lane ld/st")
        if mod.TOFFSET in modifiers:
            raise ValueError("SVE has no ld/st with 2D tile offsets")
        if mod.ISTRIDE in modifiers:
            raise ValueError("SVE has no ld/st with immediate strides")
        if mod.GSTRIDE in modifiers:
            raise ValueError("SVE has no ld/st with GP-reg strides")
        if mod.MASK in modifiers:
            raise NotImplementedError("Masked ld/st for SVE not yet implemented")
        if mod.ROW in modifiers:
            raise ValueError("SVE has no row selection ld/st")
        if mod.COL in modifiers:
            raise ValueError("SVE has no column selection ld/st")
        if mod.NT in modifiers:
            raise NotImplementedError("Non-temporals for SVE not yet implemented")

        if mod.BCAST in modifiers and self.action != opdna1_action.LOAD:
            raise ValueError("BCAST modifier is only valid for LOAD operations")
            
        if mod.VINDEX in modifiers and \
          (mod.VOFFSET in modifiers or mod.IOFFSET in modifiers):
            raise ValueError("VINDEX cannot be combined with IOFFSET/VOFFSET")

    def check_required_parameters(self, dregs: list[data_reg],
                                  modifiers: set[mod], **kwargs):
        required = []
        if mod.IOFFSET in modifiers:
            required.append("ioffset")
        if mod.VOFFSET in modifiers:
            required.append("voffset")
        if mod.GOFFSET in modifiers:
            required.append("offreg")
        if mod.STRUCT in modifiers:
            required.append("nstructs")
        if mod.VINDEX in modifiers:
            required.append("vidxreg")
            required.append("it")

        for p in required:
            if p not in kwargs:
                raise ValueError(f"Missing parameter: {p}")

        nstructs = kwargs.get("nstructs", 1)
        if mod.STRUCT not in modifiers and len(dregs) != 1:
            raise ValueError("Multiple registers provided but STRUCT modifier is missing")
        if mod.STRUCT in modifiers and len(dregs) != nstructs:
            raise ValueError(f"Number of dregs differs from nstructs ({len(dregs)} != {nstructs})")

    def get_mem_suffix(self, dt: adt) -> str:
        """ Memory view size (e.g., ld1w for 32-bit words) """
        size = adt_size(dt)
        if size == 1: return "b"
        if size == 2: return "h"
        if size == 4: return "w"
        if size == 8: return "d"
        raise ValueError(f"Unsupported SVE memory size: {size}")

    def get_element_suffix(self, dt: adt) -> str:
        """ Register view size (e.g., z0.s for 32-bit singles) """
        size = adt_size(dt)
        if size == 1: return ".b"
        if size == 2: return ".h"
        if size == 4: return ".s"
        if size == 8: return ".d"
        raise ValueError(f"Unsupported SVE element size: {size}")

    def get_addressing(self, areg: aarch64_greg,
                       modifiers: set[mod], dt: adt, **kwargs) -> str:
        if not isinstance(areg, aarch64_greg):
            raise ValueError(f"{areg} is not an aarch64_greg")

        if mod.VOFFSET in modifiers:
            return f"[{areg}, #{kwargs['voffset']}, MUL VL]"
            
        if mod.IOFFSET in modifiers:
            return f"[{areg}, #{kwargs['ioffset']}]"
            
        if mod.GOFFSET in modifiers:
            offreg = kwargs["offreg"]
            size = adt_size(dt)
            if size > 1:
                lsl = int(math.log2(size))
                return f"[{areg}, {offreg}, lsl #{lsl}]"
            return f"[{areg}, {offreg}]"

        if mod.VINDEX in modifiers:
            vidxreg = kwargs["vidxreg"]
            idx_esuf = self.get_element_suffix(kwargs["it"])
            size = adt_size(dt)
            
            # Scatter/Gather index scaling
            if size > 1:
                lsl = int(math.log2(size))
                return f"[{areg}, {vidxreg}{idx_esuf}, lsl #{lsl}]"
            return f"[{areg}, {vidxreg}{idx_esuf}]"

        return f"[{areg}]"

    def __call__(self, *, dregs: list, areg: aarch64_greg, dt: adt,
                 modifiers: set[mod], **kwargs) -> str:
                 
        if not dregs:
            raise ValueError("No dregs provided")

        # Forward scalars to AArch64 base
        if isinstance(dregs[0], (aarch64_greg, aarch64_freg)):
            return self.scalar_opdna1(dregs=dregs, areg=areg,
                                      dt=dt, modifiers=modifiers, **kwargs)

        if not all(isinstance(reg, sve_vreg) for reg in dregs):
            raise ValueError("Mixed or invalid register types for SVE vector operation")

        self.check_required_parameters(dregs=dregs, modifiers=modifiers, **kwargs)
        self.check_modifiers(modifiers)
        self.check_dt(dt)

        # 1. Resolve Suffixes
        msuf = self.get_mem_suffix(dt)
        esuf = self.get_element_suffix(dt)

        # 2. Build Base Instruction (e.g. ld1w, ld2d, ld1rw)
        nstructs = kwargs.get("nstructs", 1)
        if mod.BCAST in modifiers:
            inst = f"{self.inst_base}1r{msuf}"
        else:
            inst = f"{self.inst_base}{nstructs}{msuf}"

        # 3. Resolve Predicate (Default to p0 if not passed)
        preg = kwargs.get("preg", sve_preg(0))
        if not isinstance(preg, sve_preg):
            raise ValueError(f"{preg} is not a valid sve_preg")
        preg_str = f"{preg}/z" if self.action == opdna1_action.LOAD else f"{preg}"

        # 4. Resolve Registers and Addressing
        dregs_str = ", ".join([f"{r}{esuf}" for r in dregs])
        addressing = self.get_addressing(areg, modifiers, dt, **kwargs)

        return self.asmwrap(f"{inst} {{{dregs_str}}}, {preg_str}, {addressing}")
