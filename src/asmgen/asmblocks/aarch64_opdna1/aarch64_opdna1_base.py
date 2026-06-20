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

    def supported_dts(self) -> list[adt]:
        return [
            adt.FP64, adt.FP32, adt.FP16, adt.FP8E4M3, adt.FP8E5M2,
            adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8,
            adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8
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

    def check_required_parameters(self, dregs : list[data_reg],  modifiers: set[mod], **kwargs):

        required_extra_params = []

        if mod.IOFFSET in modifiers:
            required_extra_params.append({"ioffset"})

        if mod.GOFFSET in modifiers:
            required_extra_params.append({"offreg"})

        if mod.POSTINC in modifiers:
            required_extra_params.append({"iinc","increg"})

        for p in required_extra_params:
            params_specified = len(p.intersection(set(kwargs.keys())))
            if params_specified > 1:
                raise ValueError(f"{', '.join(sorted(p))} are mutually exclusive")
            if params_specified == 0:
                raise ValueError(f"Missing one of: {', '.join(sorted(p))}")

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

    def __call__(self, *, dregs: list, areg: aarch64_greg, dt: adt,
                 modifiers: set[mod], **kwargs) -> str:
        if not all(isinstance(r, (aarch64_greg, aarch64_freg)) for r in dregs):
            raise ValueError(
                    "aarch64_opdna1 requires scalar registers (greg/freg)")
                     
        self.check_modifiers(modifiers)
        self.check_required_parameters(dregs=dregs, modifiers=modifiers, **kwargs)
        self.check_dt(dt)

        if len(dregs) != 1:
            raise ValueError(
                    "AArch64 scalar load/store uses exactly one register.")

        dreg = dregs[0].retype(dt=dt)
        is_freg = isinstance(dreg, aarch64_freg)
        
        inst = self.get_inst_mnemonic(dt, is_freg)
        addressing = self.get_addressing(areg, modifiers, **kwargs)

        return self.asmwrap(f"{inst} {dreg}, {addressing}")
