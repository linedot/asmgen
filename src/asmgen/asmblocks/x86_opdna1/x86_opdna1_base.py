# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Base X86 loads/stores
"""

from typing import Callable, Any

from ..op import (
    opdna1,
    opdna1_modifier as mod,
    opdna1_action,
    operation_signature
)

from ...registers import asm_data_type as adt, adt_size
from ..types.avx_types import x86_greg, avx_freg

from .signatures import make_x86_64_opdna1_signatures

class x86_opdna1(opdna1):
    """
    x86_64 scalar instruction with 1 data operand and 1 address operand.
    Handles standard scalar loads/stores (movb, movq, vmovss, vmovsd)
    """

    def __init__(self, action: opdna1_action,
                 asmwrap: Callable[[str],str],
                 rpref : Callable[[str],str]):
        self.action = action
        self.asmwrap = asmwrap
        self.rpref = rpref

        self.signatures = make_x86_64_opdna1_signatures()

    def get_signatures(self) -> list[operation_signature]:
        return self.signatures

    def diagnose_failure(self, modifiers : set[mod],
                         kwargs : dict[str,Any],
                         dts : dict[str, adt]) -> list[operation_signature]:

        unsupported_mods = {
            mod.TINDEX:  (ValueError, "Base X86 has no ld/st with 2D tile offset indices"),
            mod.VINDEX:  (ValueError, "Base X86 has no ld/st with 1D vector offset indices"),
            mod.GLANE:   (ValueError, "Base X86 has no GP-reg lane ld/st"),
            mod.ILANE:   (ValueError, "Base X86 has no immediate lane ld/st"),
            mod.POSTINC: (ValueError, "Base X86 has no postinc ld/st"),
            mod.TOFFSET: (ValueError, "Base X86 has no ld/st with 2D tile offsets"),
            mod.VOFFSET: (ValueError, "Base X86 has no ld/st with vector offsets"),
            mod.ISTRIDE: (ValueError, "Base X86 has no ld/st with immediate strides"),
            mod.GSTRIDE: (ValueError, "Base X86 has no ld/st with GP-reg strides"),
            mod.STRUCT:  (ValueError, "Base X86 has no structured ld/st"),
            mod.BCAST:   (ValueError, "Base X86 has no broadcasting ld/st"),
            mod.ROW:     (ValueError, "Base X86 has no row selection ld/st"),
            mod.COL:     (ValueError, "Base X86 has no column selection ld/st"),
            mod.MASK:    (ValueError, "Base X86 has no masked ld/st"),
            mod.NT:      (NotImplementedError, "Non-temporals for Base X86 not yet implemented"),
        }
        for m, (exc_type, msg) in unsupported_mods.items():
            if m in modifiers:
                raise exc_type(msg)

        required_params = {
            mod.IOFFSET : ['ioffset'],
            mod.GOFFSET : ['offreg'],
        }
        for m, plist in required_params.items():
            for p in plist:
                if m in modifiers and p not in kwargs:
                    raise ValueError(f"{m.name} modifier requires '{p}' parameter")

    def get_scalar_suffix(self, dt: adt, is_freg: bool) -> str:
        """
        Instruction suffix for int and fp scalars

        :param dt: data type
        :param dt: if an fp register is used
        """
        fp_sufs = {2:"sh",4:"ss",8:"sd"}
        i_sufs = {1:"b",2:"w",4:"l",8:"q"}
        size = adt_size(dt)
        if is_freg:
            return fp_sufs[size]
        return i_sufs[size]

    def get_addressing(self, areg: x86_greg, modifiers: set[mod], **kwargs) -> str:
        """
        Generate addressing string
        :param areg: GP reg containing base address
        :param modifiers: operation modifiers
        :param dt: data type to use
        """

        offset = kwargs.get("ioffset", 0) if mod.IOFFSET in modifiers else 0

        pareg = self.rpref(areg)
        register_part = f"{pareg}"
        if mod.GOFFSET in modifiers:
            poffreg = self.rpref(kwargs['offreg'])
            register_part += f",{poffreg}"

        return f"{offset}({register_part})" if offset != 0 else f"({register_part})"

    def implementation(self, *, dregs: list, agreg: x86_greg, a_dt: adt,
                       modifiers: set[mod], **kwargs) -> str:

        dreg = dregs[0]

        is_freg = isinstance(dreg, avx_freg)
        base_inst = "vmov" if is_freg else "mov"
        inst = f"{base_inst}{self.get_scalar_suffix(a_dt, is_freg)}"

        addressing = self.get_addressing(agreg, modifiers, **kwargs)

        pdreg = self.rpref(dreg)

        if self.action == opdna1_action.LOAD:
            return self.asmwrap(f"{inst} {addressing}, {pdreg}")
        if self.action == opdna1_action.STORE:
            return self.asmwrap(f"{inst} {pdreg}, {addressing}")

        raise ValueError(f"Action {self.action} not valid")
