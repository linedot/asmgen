# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
from ..operations import opdna1, opdna1_modifier as mod, opdna1_action
from ...registers import asm_data_type as adt, adt_size

from ..types.aarch64_types import aarch64_greg, aarch64_freg

from typing import Callable

class aarch64_opdna1(opdna1):
    """
    AArch64 scalar instruction with 1 data operand and 1 address operand.
    Handles standard scalar loads/stores (ldr, str, ldrb, strh, etc.)
    """

    def __init__(self, action: opdna1_action, 
                 asmwrap : Callable[[str],str]):
        self.action = action
        self.asmwrap=asmwrap

    @property
    def inst_base(self):
        if self.action == opdna1_action.LOAD:
            return "ldr"
        elif self.action == opdna1_action.STORE:
            return "str"
        else:
            raise ValueError(f"Invalid action: {self.action}")

    def supported_dts(self) -> list[dict[str,adt]]:
        return [
            {'adreg': adt.FP64},
            {'adreg': adt.FP32},
            {'adreg': adt.FP16},
            {'adreg': adt.FP8E4M3},
            {'adreg': adt.FP8E5M2},
            {'adreg': adt.SINT64},
            {'adreg': adt.SINT32},
            {'adreg': adt.SINT16},
            {'adreg': adt.SINT8},
            {'adreg': adt.UINT64},
            {'adreg': adt.UINT32},
            {'adreg': adt.UINT16},
            {'adreg': adt.UINT8}
        ]

    def check_modifiers(self, modifiers: set[mod]):

        if mod.TINDEX in modifiers:
            raise ValueError(
                    "Base AArch64 has no ld/st with 2D tile offset indices")
        if mod.VINDEX in modifiers:
            raise ValueError(
                    "Base AArch64 has no ld/st with 1D vector offset indices")
        if mod.GLANE in modifiers:
            raise ValueError("Base AArch64 has no GP-reg lane ld/st")
        if mod.ILANE in modifiers:
            raise ValueError("Base AArch64 has no immediate lane ld/st")
        if mod.TOFFSET in modifiers:
            raise ValueError("Base AArch64 has no ld/st with 2D tile offsets")
        if mod.VOFFSET in modifiers:
            raise ValueError("Base AArch64 has no ld/st with vector offsets")
        if mod.ISTRIDE in modifiers:
            raise ValueError(
                    "Base AArch64 has no ld/st with immediate strides")
        if mod.GSTRIDE in modifiers:
            raise ValueError("Base AArch64 has no ld/st with GP-reg strides")
        if mod.STRUCT in modifiers:
            raise ValueError("Base AArch64 has no structured ld/st")
        if mod.BCAST in modifiers:
            raise ValueError("Base AArch64 has no broadcasting ld/st")
        if mod.MASK in modifiers:
            raise ValueError("Base AArch64 has no masked ld/st")
        if mod.ROW in modifiers:
            raise ValueError("Base AArch64 has no row selection ld/st")
        if mod.COL in modifiers:
            raise ValueError("Base AArch64 has no column selection ld/st")
        if mod.NT in modifiers:
            raise NotImplementedError("Non-temporals for Base AArch64 not yet implemented")

    def get_operand_restrictions(self, oprnd : str) -> set[operand_restriction]:
        # No restriction on any operands
        return {}

    def get_operand_restriction_value(self, op : str,
                                      modifiers : set[mod],
                                      rstr : operand_restriction) \
      -> int|set[int]|tuple[str,int]:
        raise ValueError("No restriction {rstr} on operand {op} for NEON opd3")

    def get_required_params(self, modifiers : set[mod]) -> list[str]:

        required_extra_params = []

        if mod.IOFFSET in modifiers:
            required_extra_params.append({"ioffset"})

        if mod.GOFFSET in modifiers:
            required_extra_params.append({"offreg"})

        if mod.POSTINC in modifiers:
            required_extra_params.append({"iinc","increg"})

        return required_extra_params

    def get_addressing(self,
                       areg: aarch64_greg,
                       modifiers: set[mod], **kwargs) -> str:
        if not isinstance(areg, aarch64_greg):
            raise ValueError(f"{areg} is not an aarch64_greg")
        
        base_addr = f"[{areg}]"

        if mod.POSTINC in modifiers:
            increg = kwargs.get("increg", None)
            if increg is not None:
                if not isinstance(increg, aarch64_greg):
                    raise ValueError(
                            f"Register offset increg must be an aarch64_greg")
                return f"{base_addr}, {increg}"
                
            iinc = kwargs["iinc"]
            return f"{base_addr}, #{iinc}"
            
        elif mod.IOFFSET in modifiers:
            ioffset = kwargs["ioffset"]
            if ioffset == 0:
                return base_addr
            return f"[{areg}, #{ioffset}]"
            
        elif mod.GOFFSET in modifiers: 
            offreg = kwargs["offreg"]
            if not isinstance(offreg, aarch64_greg):
                raise ValueError(
                        f"Register offset offreg must be an aarch64_greg")
            return f"[{areg}, {offreg}]"
            
        return base_addr

    def get_inst_mnemonic(self, dt: adt, is_freg: bool) -> str:
        base = self.inst_base
        
        if is_freg:
            return base 
        
        size = adt_size(dt)
        if size == 1:
            return base + "b"
        elif size == 2:
            return base + "h"
            
        return base 

    def implementation(self, *, dregs: list, agreg: aarch64_greg, a_dt: adt,
                       modifiers: set[mod], **kwargs) -> str:
        if not all(isinstance(r, (aarch64_greg, aarch64_freg)) for r in dregs):
            raise ValueError(
                    "aarch64_opdna1 requires scalar registers (greg/freg)")
                     

        if len(dregs) != 1:
            raise ValueError(
                    "AArch64 scalar load/store uses exactly one register.")

        dreg = dregs[0].retype(dt=a_dt)
        is_freg = isinstance(dreg, aarch64_freg)
        
        inst = self.get_inst_mnemonic(a_dt, is_freg)
        addressing = self.get_addressing(agreg, modifiers, **kwargs)

        return self.asmwrap(f"{inst} {dreg}, {addressing}")
