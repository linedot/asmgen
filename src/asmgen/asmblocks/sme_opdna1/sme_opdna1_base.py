# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
import math
from typing import Callable

from ..operations import opdna1_modifier as mod, opdna1_action, opdna1, operand_restriction
from ...registers import asm_data_type as adt, adt_size, data_reg

from ..types.aarch64_types import aarch64_greg
from ..types.sve_types import sve_preg
from ..types.sme_types import sme_treg
from ..sve_opdna1 import sve_opdna1

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

    @property
    def inst_base(self):
        return "ld" if self.action == opdna1_action.LOAD else "st"

    def supported_dts(self) -> list[adt]:
        return self.sve_opdna1.supported_dts()

    def check_modifiers(self, modifiers: set[mod]):
        if mod.ROW in modifiers and mod.COL in modifiers:
            raise ValueError("ROW and COL modifiers are mutually exclusive")

        if mod.NT in modifiers and (mod.ROW in modifiers or mod.COL in modifiers):
            raise ValueError("Non-temporal strided operations cannot be combined with tile slice modifiers (ROW/COL)")

        if mod.TINDEX in modifiers:
            raise ValueError("SME uses ROW/COL modifiers instead of TINDEX")
        if mod.GLANE in modifiers:
            raise ValueError("SME has no GP-reg lane ld/st")
        if mod.ILANE in modifiers:
            raise ValueError("SME has no immediate lane ld/st")
        if mod.POSTINC in modifiers:
            raise ValueError("SME native instructions do not support POSTINC")
        if mod.ISTRIDE in modifiers:
            raise ValueError("SME native instructions do not support immediate strides")
        if mod.GSTRIDE in modifiers:
            raise ValueError("SME native instructions do not support GP-reg strides")
        if mod.STRUCT in modifiers:
            raise ValueError("SME uses multiple discrete registers for strided loads, not STRUCT")
        if mod.MASK in modifiers:
            raise NotImplementedError("Masked ld/st for SME not yet implemented")
        if mod.BCAST in modifiers:
            raise ValueError("SME native instructions do not support BCAST")
        if mod.VINDEX in modifiers:
            raise ValueError("SME native instructions do not support vector indices (Gathers/Scatters)")
        if mod.TOFFSET in modifiers:
            raise ValueError("SME native instructions do not support 2D tile offsets (use ROW/immrow or COL/immcol instead)")

        if mod.ROW in modifiers or mod.COL in modifiers:
            if mod.VOFFSET in modifiers or mod.IOFFSET in modifiers:
                raise ValueError("SME tile slice operations only support scalar+scalar (GOFFSET) memory addressing")

        addr_mods = {mod.IOFFSET, mod.VOFFSET, mod.GOFFSET}.intersection(modifiers)
        if len(addr_mods) > 1:
            raise ValueError(f"Mutually exclusive addressing modifiers used: {addr_mods}")

    def get_required_params(self, modifiers: set[mod]) -> list[set[str]]:
        required = []
        
        if mod.ROW in modifiers:
            required.append({"rowreg"})
            required.append({"immrow"})
        if mod.COL in modifiers:
            required.append({"colreg"})
            required.append({"immcol"})
        if mod.IOFFSET in modifiers:
            required.append({"ioffset"})
        if mod.VOFFSET in modifiers:
            required.append({"voffset"})
        if mod.GOFFSET in modifiers:
            required.append({"offreg"})

        return required


    def get_operand_restrictions(self, oprnd : str) -> set[operand_restriction]:
        # No restriction on any operands
        return {
            'bdreg' : operand_restriction.IDXOTHERPLUSNMOD,
            'cdreg' : operand_restriction.IDXOTHERPLUSNMOD,
            'ddreg' : operand_restriction.IDXOTHERPLUSNMOD,
            'rowreg': operand_restriction.IDXMIN,
            'rowreg': operand_restriction.IDXMAX,
            'colreg': operand_restriction.IDXMIN,
            'colreg': operand_restriction.IDXMAX,
        }

    def get_operand_restriction_value(self, op : str,
                                      rstr : operand_restriction) \
      -> int|set[int]|tuple[str,int]:

        if op in {'bdreg', 'cdreg', 'ddreg'} and \
          rstr == operand_restriction.IDXOTHERPLUSNMOD:
            return (chr(ord(op[0])+1)+'dreg', 1, 32)

        if op in {'rowreg', 'colreg'} and \
          rstr == operand_restriction.IDXMAX:
            return 15
        if op in {'rowreg', 'colreg'} and \
          rstr == operand_restriction.IDXMIN:
            return 12

        raise ValueError("No restriction {rstr} on operand {op} for SVE opdna1")

    def get_mem_suffix(self, dt: adt) -> str:
        size = adt_size(dt)
        if size == 1: return "b"
        if size == 2: return "h"
        if size == 4: return "w"
        if size == 8: return "d"
        raise ValueError(f"Unsupported SME memory size: {size}")

    def get_element_suffix(self, dt: adt) -> str:
        size = adt_size(dt)
        if size == 1: return ".b"
        if size == 2: return ".h"
        if size == 4: return ".s"
        if size == 8: return ".d"
        raise ValueError(f"Unsupported SME element size: {size}")

    def get_addressing(self, areg: aarch64_greg, modifiers: set[mod], dt: adt, **kwargs) -> str:
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

    def implementation(self, *, dregs: list, agreg: aarch64_greg, a_dt: adt,
                       modifiers: set[mod], **kwargs) -> str:
                 
        if not dregs:
            raise ValueError("No dregs provided")

        # --- ROUTING LOGIC ---
        # If it's not a Tile Register AND it's not a Non-Temporal instruction, SVE handles it.
        if not isinstance(dregs[0], sme_treg) and mod.NT not in modifiers:
            return self.sve_opdna1(dregs=dregs, areg=agreg, dt=a_dt,
                                   modifiers=modifiers, **kwargs)

        msuf = self.get_mem_suffix(a_dt)
        esuf = self.get_element_suffix(a_dt)
        addressing = self.get_addressing(agreg, modifiers, a_dt, **kwargs)

        # SME specific checks not handled by default checks
        if mod.NT in modifiers:
            if len(dregs) not in [1, 2, 4]:
                raise ValueError(("SME2 non-temporal operations require 1, 2, or"
                                 f" 4 vector registers, got {len(dregs)}"))
                
        if mod.ROW in modifiers or mod.COL in modifiers:
            if len(dregs) != 1:
                raise ValueError(("SME tile slice operations accept exactly one "
                                 f"sme_treg, got {len(dregs)}"))

        # 1. Tile Slice (ZA)
        if isinstance(dregs[0], sme_treg):
            if len(dregs) != 1:
                raise ValueError("SME tile slice operations accept exactly one sme_treg")
            if not (mod.ROW in modifiers or mod.COL in modifiers):
                raise ValueError("Tile slice requires ROW or COL modifier")

            hv = "h" if mod.ROW in modifiers else "v"
            idx_reg = kwargs["rowreg"] if mod.ROW in modifiers else kwargs["colreg"]
            slice_imm = kwargs["immrow"] if mod.ROW in modifiers else kwargs["immcol"]

            if idx_reg.idx < 12 or idx_reg.idx > 15:
                raise ValueError(f"row/col slice greg {idx_reg} is not w12-w15")
            
            # e.g., za0h.d[w12, 0]
            dregs_str = f"{dregs[0]}{hv}{esuf}[{idx_reg.get_wreg()}, {slice_imm}]"
            inst = f"{self.inst_base}1{msuf}"
            
            preg = kwargs.get("preg", sve_preg(0))
            preg_str = f"{preg}/z" if self.action == opdna1_action.LOAD else f"{preg}"

        elif mod.NT in modifiers:
            # Validate strided Z-register tuples for SME2
            if len(dregs) == 2:
                if dregs[1].idx != dregs[0].idx + 8:
                    raise ValueError(("SME2 2-register operations require a register"
                                     f" stride of 8 (e.g. Z0, Z8). Got z{dregs[0].idx},"
                                     f" z{dregs[1].idx}"))
                if (dregs[0].idx % 16) >= 8:
                    raise ValueError(("First register of SME2 2-register tuple must be"
                                     f"Z0-Z7 or Z16-Z23. Got z{dregs[0].idx}"))
            
            elif len(dregs) == 4:
                for i in range(1, 4):
                    if dregs[i].idx != dregs[0].idx + i * 4:
                        raise ValueError(("SME2 4-register operations require a register"
                                          "stride of 4 (e.g. Z0, Z4, Z8, Z12)."))
                if (dregs[0].idx % 16) >= 4:
                    raise ValueError(("First register of SME2 4-register tuple must be Z0-Z3"
                                     f" or Z16-Z19. Got z{dregs[0].idx}"))

            dregs_str = ", ".join([f"{r}{esuf}" for r in dregs])
            inst = f"{self.inst_base}nt1{msuf}"
            
            # Predicate-as-counter (PN8-PN15) validation and string generation
            preg = kwargs.get("preg", sve_preg(8, is_pn=True)) 
            preg_str = f"{preg}/z" if self.action == opdna1_action.LOAD else f"{preg}"
        else:
            raise ValueError("Unhandled SME instruction configuration")

        return self.asmwrap(f"{inst} {{{dregs_str}}}, {preg_str}, {addressing}")
