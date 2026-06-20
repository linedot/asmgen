# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
from ..operations import opdna1, opdna1_modifier as mod, opdna1_action
from ...registers import asm_data_type as adt, adt_size
from ..types.avx_types import x86_greg, avx_freg

from typing import Callable

class x86_opdna1(opdna1):
    """
    x86-64 scalar instruction with 1 data operand and 1 address operand.
    Handles standard scalar loads/stores (movb, movq, vmovss, vmovsd)
    """

    def __init__(self, action: opdna1_action,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        self.action = action
        self.asmwrap = asmwrap
        self.rpref = rpref

    def supported_dts(self) -> list[adt]:
        return [
            adt.FP64, adt.FP32, adt.FP16, adt.BF16,
            adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8,
            adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8
        ]

    def check_modifiers(self, modifiers: set[mod]):
        if mod.TINDEX in modifiers:
            raise ValueError("Base X86 has no ld/st with 2D tile offset indices")
        if mod.VINDEX in modifiers:
            raise ValueError("Base X86 has no ld/st with 1D vector offset indices")
        if mod.GLANE in modifiers:
            raise ValueError("Base X86 has no GP-reg lane ld/st")
        if mod.ILANE in modifiers:
            raise ValueError("Base X86 has no immediate lane ld/st")
        if mod.POSTINC in modifiers:
            raise ValueError("Base X86 has no postinc ld/st form")
        if mod.TOFFSET in modifiers:
            raise ValueError("Base X86 has no ld/st with 2D tile offsets")
        if mod.VOFFSET in modifiers:
            raise ValueError("Base X86 has no ld/st with vector offsets")
        if mod.ISTRIDE in modifiers:
            raise ValueError("Base X86 has no ld/st with immediate strides")
        if mod.GSTRIDE in modifiers:
            raise ValueError("Base X86 has no ld/st with gp-reg strides")
        if mod.STRUCT in modifiers:
            raise ValueError("Base X86 has no structured ld/st")
        if mod.BCAST in modifiers:
            raise ValueError("Base X86 has no broadcast ld/st")
        if mod.MASK in modifiers:
            raise ValueError("Base X86 has no masked ld/st")

    def check_required_parameters(self, dregs: list, modifiers: set[mod], **kwargs):
        if mod.IOFFSET in modifiers and "ioffset" not in kwargs:
            raise ValueError("Missing parameter: ioffset")

    def get_scalar_suffix(self, dt: adt, is_freg: bool) -> str:
        size = adt_size(dt)
        if is_freg:
            if size == 2: return "sh"
            if size == 4: return "ss"
            if size == 8: return "sd"
            raise ValueError(f"Unsupported x86 scalar float size: {size}")
            
        if size == 1: return "b"
        if size == 2: return "w"
        if size == 4: return "l"
        if size == 8: return "q"
        raise ValueError(f"Unsupported x86 scalar integer size: {size}")

    def get_addressing(self, areg: x86_greg, modifiers: set[mod], **kwargs) -> str:
        if not isinstance(areg, x86_greg):
            raise ValueError(f"{areg} is not an x86_greg")
            
        offset = kwargs.get("ioffset", 0) if mod.IOFFSET in modifiers else 0
        pareg = self.rpref(areg)
        return f"{offset}({pareg})" if offset != 0 else f"({pareg})"

    def __call__(self, *, dregs: list, areg: x86_greg, dt: adt,
                 modifiers: set[mod], **kwargs) -> str:
                 
        if len(dregs) != 1:
            raise ValueError("x86 scalar load/store uses exactly one register")

        dreg = dregs[0]
        if not isinstance(dreg, (x86_greg, avx_freg)):
            raise ValueError("x86_opdna1 requires scalar registers (x86_greg or avx_freg)")

        self.check_modifiers(modifiers)
        self.check_required_parameters(dregs, modifiers, **kwargs)
        self.check_dt(dt)

        is_freg = isinstance(dreg, avx_freg)
        base_inst = "vmov" if is_freg else "mov"
        inst = f"{base_inst}{self.get_scalar_suffix(dt, is_freg)}"
        
        addressing = self.get_addressing(areg, modifiers, **kwargs)

        pdreg = self.rpref(dreg)

        if self.action == opdna1_action.LOAD:
            return self.asmwrap(f"{inst} {addressing}, {pdreg}")
        else: # STORE
            return self.asmwrap(f"{inst} {pdreg}, {addressing}")
