# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RVV 1.0 and 0.7.1 opdna1 base
"""

from typing import Any,Callable

from ...registers import (
    asm_data_type as adt,
    adt_size,
    data_reg,
    greg_base
)

from ..op import (
    opdna1,
    opdna1_modifier as mod,
    opdna1_action,
    operation_signature
)

from ..riscv64_opdna1.riscv64_opdna1_base import riscv64_opdna1

from ..types.rvv_types import rvv_vreg
from ..types.riscv64_types import riscv64_greg,riscv64_freg


from .signatures import make_rvv_opdna1_signatures

class rvv_opdna1(opdna1):
    """
    RVV instruction with 1 data operand and 1 address operand

    Abstraction for loads/stores (maybe also prefetches)
    """

    def __init__(self, action : opdna1_action,
                 asmwrap : Callable[[str],str],
                 lmul_getter :Callable[[],int]):
        self.action = action
        self.asmwrap = asmwrap
        self.get_lmul = lmul_getter

        self.scalar_opdna1 = riscv64_opdna1(action=action, asmwrap=asmwrap)

        self.signatures = make_rvv_opdna1_signatures(self.get_lmul)
        self.signatures.extend(self.scalar_opdna1.get_signatures())

    def get_signatures(self) -> list[operation_signature]:
        return self.signatures

    @property
    def inst_base(self):
        """
        Instruction base
        """
        if self.action  == opdna1_action.LOAD:
            return "vl"
        if self.action == opdna1_action.STORE:
            return "vs"
        raise ValueError(f"Invalid action: {self.action}")

    # I explicitly want it this way
    # pylint: disable-next=too-many-branches
    def diagnose_failure(self, modifiers : set[mod],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):
        # Unsupported modifiers
        if mod.TINDEX in modifiers:
            raise ValueError("RVV has no ld/st with 2D tile offset indices")
        if mod.GLANE in modifiers:
            raise ValueError("RVV has no GP-reg lane ld/st")
        if mod.ILANE in modifiers:
            raise ValueError("RVV has no immediate lane ld/st")
        if mod.POSTINC in modifiers:
            raise ValueError("RVV has no postinc ld/st form")
        if mod.TOFFSET in modifiers:
            raise ValueError("RVV has no ld/st with 2D tile offsets")
        if mod.VOFFSET in modifiers:
            raise ValueError("RVV has no ld/st with vector offsets")
        if mod.IOFFSET in modifiers:
            raise ValueError("RVV has no ld/st with immediate offsets")
        if mod.ISTRIDE in modifiers:
            raise ValueError("RVV has no ld/st with immediate strides")
        if mod.MASK in modifiers:
            raise NotImplementedError("Masked ld/st for RVV not yet implemented")
        if mod.ROW in modifiers:
            raise ValueError("RVV has no row selection ld/st")
        if mod.COL in modifiers:
            raise ValueError("RVV has no column selection ld/st")
        if mod.NT in modifiers:
            raise NotImplementedError("Non-temporals for RVV not yet implemented")

        if (mod.GSTRIDE in modifiers) and (mod.VINDEX in modifiers):
            raise ValueError("mod.GSTRIDE and mod.VINDEX are mutually exclusive")

        # Missing args
        if mod.VINDEX in modifiers:
            if "vidxreg" not in kwargs:
                raise ValueError("Missing one of these parameters: vidxreg")
            if "it" not in kwargs:
                raise ValueError("Missing one of these parameters: it")

        if mod.STRUCT in modifiers:
            if "nstructs" not in kwargs:
                raise ValueError("Missing one of these parameters: nstructs")

        if mod.GSTRIDE in modifiers and "streg" not in kwargs:
            raise ValueError("Missing one of these parameters: streg")


    def get_instruction(self, base : str,
                        modifiers: set[mod], dt : adt, **kwargs) -> str:
        """
        Constructs the instruction mnemonic

        :param base: instruction base
        :param modifiers: operation modifiers
        :param dt: data type to use

        :return: string containing the mnemonic
        """

        inst_name = base

        # vl/vs
        # if mod.GSTRIDE: +s
        # if mod.VINDEX: +ux
        # if mod.STRUCT: +seg +{nf}
        # +e
        # if mod.VINDEX: +i
        # +{eew}

        if mod.GSTRIDE in modifiers:
            inst_name += "s"
        if mod.VINDEX in modifiers:
            inst_name += "ux" # only unordered.
        if mod.STRUCT in modifiers:
            nf = kwargs["nstructs"]
            inst_name += f"seg{nf}"
        inst_name += "e"
        if mod.VINDEX in modifiers:
            inst_name += "i"

        inst_name += str(adt_size(dt)*8)

        inst_name += ".v"


        return inst_name

    def get_addressing(self, areg : riscv64_greg,
                       modifiers: set[mod], **kwargs) -> str:
        """
        Constructs the addressing string

        :param areg: address register
        :param modifiers: operation modifiers
        
        :return: string containing the addressing
        """
        if not isinstance(areg, riscv64_greg):
            raise ValueError(f"{areg} is not a riscv64_greg")

        base_addr = f"({areg})"

        if mod.GSTRIDE in modifiers:
            streg = kwargs["streg"]
            if not isinstance(streg, riscv64_greg):
                raise ValueError(f"{streg} is not a riscv64_greg")
            return f"{base_addr}, {streg}"

        if mod.VINDEX in modifiers:
            vidxreg = kwargs["vidxreg"]
            if not isinstance(vidxreg, rvv_vreg):
                raise ValueError(f"{vidxreg} is not a rvv_vreg")
            return f"{base_addr}, {vidxreg}"

        return base_addr


    def implementation(self, *, dregs : list[data_reg],
                       agreg : greg_base, a_dt : adt,
                       modifiers : set[mod], **kwargs) -> str:

        if not dregs:
            raise ValueError("No dregs provided")

        # If scalar registers are passed, forward to base RISC-V
        if isinstance(dregs[0], (riscv64_greg, riscv64_freg)):
            return self.scalar_opdna1(
                    dregs=dregs, areg=agreg, dt=a_dt,
                    modifiers=modifiers, **kwargs)

        inst = self.get_instruction(self.inst_base, modifiers, a_dt, **kwargs)
        addressing = self.get_addressing(agreg, modifiers, **kwargs)


        dreg_str = str(dregs[0])
        return self.asmwrap(f"{inst} {dreg_str}, {addressing}")
