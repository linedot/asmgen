# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
from ..operations import opdna1,opdna1_modifier as mod, opdna1_action
from ...registers import asm_data_type as adt, adt_size
from ..types.riscv64_types import riscv64_greg,riscv64_freg

from typing import Callable


class riscv64_opdna1(opdna1):
    """
    RISC-V 64 bit instructions with n data operand and 1 address operand

    Absraction for loads/stores (maybe also prefetches)
    """

    def __init__(self, action : opdna1_action,
                 asmwrap : Callable[[str],str]):
        self.action = action
        self.asmwrap = asmwrap

    @property
    def inst_base(self):
        if self.action  == opdna1_action.LOAD:
            return "l"
        elif self.action == opdna1_action.STORE:
            return "s"
        else:
            raise ValueError(f"Invalid action: {self.action}")

    def supported_dts(self) -> list[adt]:

        sup_dts = [adt.FP64, adt.FP32, adt.FP16,
                   adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8,
                   adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]

        return [{'adreg': dt} for dt in sup_dts]

    def get_dt_suffix(self, dt : adt):
        size_map = {1: "b", 2: "h", 4: "w", 8: "d", 16: "q"}
        return size_map[adt_size(dt)]

    def check_modifiers(self, modifiers : set[opdna1_modifier]):

        if mod.TINDEX in modifiers:
            raise ValueError("RISC-V +D/F has no ld/st with 2D tile offset indices")
        if mod.VINDEX in modifiers:
            raise ValueError("RISC-V +D/F has no ld/st with 1D vector offset indices")
        if mod.GLANE in modifiers:
            raise ValueError("RISC-V +D/F has no GP-reg lane ld/st")
        if mod.ILANE in modifiers:
            raise ValueError("RISC-V +D/F has no immediate lane ld/st")
        if mod.POSTINC in modifiers:
            raise ValueError("RISC-V +D/F has no postinc ld/st form")
        if mod.TOFFSET in modifiers:
            raise ValueError("RISC-V +D/F has no ld/st with 2D tile offsets")
        if mod.VOFFSET in modifiers:
            raise ValueError("RISC-V +D/F has no ld/st with vector offsets")
        if mod.ISTRIDE in modifiers:
            raise ValueError("RISC-V +D/F has no ld/st with immediate strides")
        if mod.GSTRIDE in modifiers:
            raise ValueError("RISC-V +D/F has no ld/st with GP-reg strides")
        if mod.STRUCT in modifiers:
            raise ValueError("RISC-V +D/F has no structured ld/st")
        if mod.MASK in modifiers:
            raise ValueError("RISC-V +D/F has no masked ld/st")
        if mod.ROW in modifiers:
            raise ValueError("RISC-V +D/F has no row selection ld/st")
        if mod.COL in modifiers:
            raise ValueError("RISC-V +D/F has no column selection ld/st")
        if mod.NT in modifiers:
            raise NotImplementedError("Non-temporals for RISC-V +D/F not yet implemented")
        
    def get_addressing(self, areg: riscv64_greg, modifiers: set[mod], **kwargs) -> str:
        if not isinstance(areg, riscv64_greg):
            raise ValueError(f"{areg} is not a riscv64_greg")
            
        offset = kwargs.get("ioffset", 0) if mod.IOFFSET in modifiers else 0
        return f"{offset}({areg})"

    def get_required_params(self, modifiers: set[mod]) -> list[set[str]]:

        required_extra_params = []

        if mod.IOFFSET in modifiers:
            required_extra_params.append({"ioffset"})

        return required_extra_params

    def get_operand_restrictions(self, oprnd : str) -> set[operand_restriction]:
        # No restriction on any operands
        return {}

    def get_operand_restriction_value(self, op : str,
                                      rstr : operand_restriction) \
      -> int|set[int]|tuple[str,int]:
        raise ValueError("No restriction {rstr} on operand {op} for RISC-V +D/F opd3")


    def implementation(self, *, dregs : list[data_reg], agreg : greg_type, a_dt : adt,
                       modifiers : set[opdna1_modifier], **kwargs) -> str:


        if len(dregs) != 1:
            raise ValueError("RISC-V +D/F load/store uses one and only one register")

        dreg = dregs[0]

        if not isinstance(dreg, (riscv64_freg, riscv64_greg)):
            raise ValueError("{dreg} is neither a riscv64_freg nor a riscv64_greg")

        inst = self.inst_base
        if isinstance(dreg, riscv64_freg):
            inst = "f"+inst


        inst += self.get_dt_suffix(a_dt)
        addressing = self.get_addressing(agreg, modifiers, **kwargs)

        return self.asmwrap(f"{inst} {dreg}, {addressing}")


