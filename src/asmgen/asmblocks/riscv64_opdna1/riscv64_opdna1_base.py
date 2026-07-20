# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RISCV64 +D/F opdna1 operations
"""

from typing import Callable

from ...registers import (
    asm_data_type as adt,
    adt_size,
    data_reg,
    greg_base
)

from ..op import opdna1,opdna1_modifier as mod, opdna1_action
from ..op import operation_signature
from ..types.riscv64_types import riscv64_greg,riscv64_freg


from .signatures import make_riscv64_opdna1_signatures


class riscv64_opdna1(opdna1):
    """
    RISC-V 64 bit instructions with n data operand and 1 address operand

    Absraction for loads/stores (maybe also prefetches)
    """

    def __init__(self, action : opdna1_action,
                 asmwrap : Callable[[str],str]):
        self.action = action
        self.asmwrap = asmwrap

        self.signatures = make_riscv64_opdna1_signatures()

    @property
    def inst_base(self):
        """
        Instruction mnemonic base string
        """
        if self.action  == opdna1_action.LOAD:
            return "l"
        if self.action == opdna1_action.STORE:
            return "s"
        raise ValueError(f"Invalid action: {self.action}")

    def get_dt_suffix(self, dt : adt):
        """
        Instruction suffix based on data type

        :param dt: element data type
        """
        size_map = {1: "b", 2: "h", 4: "w", 8: "d", 16: "q"}
        return size_map[adt_size(dt)]

    def get_addressing(self, areg: riscv64_greg, modifiers: set[mod], **kwargs) -> str:
        """
        Constructs the addressing string

        :param areg: address register
        :param modifiers: operation modifiers
        
        :return: string containing the addressing
        """
        if not isinstance(areg, riscv64_greg):
            raise ValueError(f"{areg} is not a riscv64_greg")

        offset = kwargs.get("ioffset", 0) if mod.IOFFSET in modifiers else 0
        return f"{offset}({areg})"

    def get_signatures(self) -> list[operation_signature]:
        return self.signatures

    def implementation(self, *, dregs : list[data_reg], agreg : greg_base, a_dt : adt,
                       modifiers : set[mod], **kwargs) -> str:


        dreg = dregs[0]

        inst = self.inst_base
        if isinstance(dreg, riscv64_freg):
            inst = "f"+inst


        inst += self.get_dt_suffix(a_dt)
        addressing = self.get_addressing(agreg, modifiers, **kwargs)

        return self.asmwrap(f"{inst} {dreg}, {addressing}")
