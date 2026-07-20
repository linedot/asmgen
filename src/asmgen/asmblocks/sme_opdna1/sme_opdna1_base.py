# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
opdna1 base class for SME
"""
import math
from typing import Callable, Any

from ..op import (
    opdna1,
    opdna1_modifier as mod,
    opdna1_action,
    operation_signature
)

from ...registers import asm_data_type as adt, adt_size

from ..types.aarch64_types import aarch64_greg
from ..types.sme_types import sme_treg
from ..sve_opdna1 import sve_opdna1

from .signatures import make_sme_opdna1_signatures

class sme_opdna1(opdna1):
    """
    AArch64 SME instruction generator.
    Handles ZA tile slices (sme_treg) and SME2 Non-Temporal loads/stores.
    Routes all other operations to sve_opdna1.
    """

    def __init__(self, action: opdna1_action, asmwrap: Callable[[str],str]):
        self.action = action
        self.asmwrap = asmwrap

        self.sve_opdna1 = sve_opdna1(action=action, asmwrap=asmwrap)

        self.signatures = make_sme_opdna1_signatures()
        self.signatures.extend(self.sve_opdna1.get_signatures())

    def get_signatures(self) -> list[operation_signature]:
        return self.signatures

    @property
    def inst_base(self):
        """
        Instruction string base root
        """
        return "ld" if self.action == opdna1_action.LOAD else "st"

    def diagnose_failure(self, modifiers: set[mod],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):

        unsupported_mods = {
            mod.BCAST:   (ValueError, "SME has no broadcasting ld"),
            mod.GLANE:   (ValueError, "SME has no GP-reg lane ld/st"),
            mod.GSTRIDE: (ValueError, "SME has no ld/st with GP-reg strides"),
            mod.ILANE:   (ValueError, "SME has no immediate lane ld/st"),
            mod.IOFFSET: (ValueError, "SME has no ld/st with immediate element offsets"),
            mod.ISTRIDE: (ValueError, "SME has no ld/st with immediate strides"),
            mod.POSTINC: (ValueError, "SME has no postinc ld/st"),
            mod.VINDEX:  (ValueError, "SME has no gather/scatter"),
            mod.TINDEX:  (ValueError, "SME has no ld/st with 2D tile offset indices"),
            mod.TOFFSET: (ValueError, "SME has no ld/st with 2D tile offsets"),
        }

        for m, (exc_type, msg) in unsupported_mods.items():
            if m in modifiers:
                raise exc_type(msg)
        if mod.ROW in modifiers and mod.COL in modifiers:
            raise ValueError("ROW and COL modifiers are mutually exclusive")

        if mod.NT in modifiers and (mod.ROW in modifiers or mod.COL in modifiers):
            raise ValueError(("Non-temporal strided operations cannot be combined "
                              "with tile slice modifiers (ROW/COL)"))

        if mod.ROW in modifiers or mod.COL in modifiers:
            if mod.VOFFSET in modifiers or mod.IOFFSET in modifiers:
                raise ValueError(("SME tile slice operations only support scalar+scalar "
                                 "(GOFFSET) memory addressing"))

        addr_mods = {mod.IOFFSET, mod.VOFFSET, mod.GOFFSET}.intersection(modifiers)
        if len(addr_mods) > 1:
            raise ValueError(f"Mutually exclusive addressing modifiers used: {addr_mods}")


        required_params = {
            mod.ROW : ['rowreg','immrow'],
            mod.COL : ['colreg','immcol'],
            mod.STRUCT : ['nstructs'],
            mod.IOFFSET : ['ioffset'],
            mod.VOFFSET : ['voffset'],
            mod.GOFFSET : ['offreg'],
        }
        for m, plist in required_params.items():
            for p in plist:
                if m in modifiers and p not in kwargs:
                    raise ValueError(f"{m.name} modifier requires '{p}' parameter")

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
        raise ValueError(f"Unsupported SME memory size: {size}")

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
        raise ValueError(f"Unsupported SME element size: {size}")

    def get_addressing(self, areg: aarch64_greg, modifiers: set[mod], dt: adt, **kwargs) -> str:
        """
        Generate addressing string
        :param areg: GP reg containing base address
        :param modifiers: operation modifiers
        :param dt: data type to use
        """
        # Standardizes addressing for SME-specific instructions
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

        return f"[{areg}]"

    # Inlining any params or breaking the method up IMHO doesn't impove readability
    # pylint: disable-next=too-many-locals
    def implementation(self, *, dregs: list, agreg: aarch64_greg, a_dt: adt,
                       modifiers: set[mod], **kwargs) -> str:

        # --- ROUTING LOGIC ---
        # If it's not a Tile Register AND it's not a Non-Temporal instruction, SVE handles it.
        if not isinstance(dregs[0], sme_treg) and mod.NT not in modifiers:
            return self.sve_opdna1(dregs=dregs, areg=agreg, dt=a_dt,
                                   modifiers=modifiers, **kwargs)

        msuf = self.get_mem_suffix(a_dt)
        esuf = self.get_element_suffix(a_dt)
        addressing = self.get_addressing(agreg, modifiers, a_dt, **kwargs)

        # 1. Tile Slice (ZA)
        if isinstance(dregs[0], sme_treg):

            hv = "h" if mod.ROW in modifiers else "v"
            idx_reg = kwargs["rowreg"] if mod.ROW in modifiers else kwargs["colreg"]
            slice_imm = kwargs["immrow"] if mod.ROW in modifiers else kwargs["immcol"]

            # e.g., za0h.d[w12, 0]
            dregs_str = f"{dregs[0]}{hv}{esuf}[{idx_reg.get_wreg()}, {slice_imm}]"
            inst = f"{self.inst_base}1{msuf}"

            preg = kwargs["amreg"]
            preg_str = f"{preg}/z" if self.action == opdna1_action.LOAD else f"{preg}"

        elif mod.NT in modifiers:
            dregs_str = ", ".join([f"{r}{esuf}" for r in dregs])
            inst = f"{self.inst_base}nt1{msuf}"

            # Predicate-as-counter (PN8-PN15) validation and string generation
            preg = kwargs["amreg"]
            preg_str = f"{preg}/z" if self.action == opdna1_action.LOAD else f"{preg}"
        else:
            raise ValueError("Unhandled SME instruction configuration")

        return self.asmwrap(f"{inst} {{{dregs_str}}}, {preg_str}, {addressing}")
