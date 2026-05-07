# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RVV 1.0 and 0.7.1 opdna1 base
"""
from ...registers import asm_data_type as adt,adt_size

from ..operations import opdna1_modifier as mod,opdna1_action

from ..riscv64_opdna1.riscv64_opdna1_base import riscv64_opdna1

from ..types.rvv_types import rvv_vreg
from ..types.riscv64_types import riscv64_greg

class rvv_opdna1(riscv64_opdna1):
    """
    RVV instruction with 1 data operand and 1 address operand

    Abstraction for loads/stores (maybe also prefetches)
    """

    def __init__(self, action : opdna1_action,
                 lmul_getter :Callable[[],int]):
        self.action = action
        self.get_lmul = lmul_getter

    @property
    def inst_base(self):
        if self.action  == opdna1_action.LOAD:
            return "vl"
        elif self.action == opdna1_action.STORE:
            return "vs"
        else:
            raise ValueError(f"Invalid action: {self.action}")

    def supported_dts(self) -> list[adt]:
       
        # TODO: more types
        return [adt.FP64, adt.FP32, adt.FP16]

    def check_modifiers(self, modifiers : set[opdna1_modifier]):

        # Unsupported
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

        #TODO: invalid combinations
        if (mod.GSTRIDE in modifiers) and (mod.VINDEX in modifiers):
            raise ValueError("mod.GSTRIDE and mod.VINDEX are mutually exclusive")


    def check_required_parameters(self, dregs : list[data_reg],  modifiers: set[mod], **kwargs):

        required_extra_params = []

        if mod.VINDEX in modifiers:
            required_extra_params.append("vidxreg")
        if mod.STRUCT in modifiers:
            required_extra_params.append("nstructs")
        if mod.GSTRIDE in modifiers:
            required_extra_params.append("streg")

        for p in required_extra_params:
            if p not in kwargs:
                raise ValueError(f"Missing parameter: {p}")

        if mod.STRUCT in modifiers:
            # Need to check but if it raises here, it'd raise later anyway
            nstructs = kwargs["nstructs"]
            if nstructs != len(dregs):
                raise ValueError(f"{nstructs} nstructs specified but only {len(dregs)} dregs given")


    def get_instruction(self, base : str,
                        modifiers: set[mod], dt : adt, **kwargs):

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

    def get_addressing(self, base : str, modifiers: set[mod], dt : adt, **kwargs):
        pass

    def get_addressing(self, areg : riscv64_greg, modifiers: set[mod], **kwargs) -> str:
        if not isinstance(areg, riscv64_greg):
            raise ValueError(f"{areg} is not a riscv64_greg")

        base_addr = f"({areg})"
        
        if mod.GSTRIDE in modifiers:
            streg = kwargs["streg"]
            if not isinstance(streg, riscv64_greg):
                raise ValueError(f"{streg} is not a riscv64_greg")
            return f"{base_addr}, {streg}"
        elif mod.VINDEX in modifiers:
            vidxreg = kwargs["vidxreg"]
            if not isinstance(vidxreg, rvv_vreg):
                raise ValueError(f"{vidxreg} is not a riscv64_greg")
            return f"{base_addr}, {vidxreg}"
            
        return base_addr


    def __call__(self, *, dregs : list[data_reg], areg : greg_type, dt : adt,
                 modifiers : set[mod], **kwargs) -> str:

        self.check_modifiers(modifiers)
        self.check_dt(dt)
        self.check_required_parameters(dregs, modifiers, **kwargs)
        inst = self.get_instruction(self.inst_base, modifiers, dt, **kwargs)
        addressing = self.get_addressing(areg, modifiers, **kwargs)


        v = rvv_vreg(0)

        # MUST be vregs
        if not all([isinstance(reg, rvv_vreg) for reg in dregs]):
            raise ValueError("RVV opdna1: All dregs must be vregs")

        for i in range(1, len(dregs)):
            if dregs[i].reg_idx != dregs[i-1].reg_idx + self.get_lmul():
                raise ValueError(
                    f"Segmented registers must be consecutive. "
                    f"Found {dregs[i-1]} followed by {dregs[i]}."
                )


        dreg_str = str(dregs[0])
        return f"{inst} {dreg_str}, {addressing}"
        
        raise NotImplementedError(self.NIE_MESSAGE)
