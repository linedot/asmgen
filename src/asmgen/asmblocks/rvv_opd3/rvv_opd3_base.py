# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RVV 1.0 and 0.7.1 opd3 base
"""
from abc import abstractmethod
from typing import Callable,Any

from ...registers import (
    asm_data_type as adt,
    adt_size,
    adt_is_float,adt_is_int,
    adt_is_signed,adt_is_unsigned,
    data_reg
)
from ..op import (
    opd3,
    opd3_modifier as mod,
    operation_signature
)
from ...util import NIE_MESSAGE


from .signatures import make_rvv_opd3_signatures

class rvv_opd3_base(opd3):
    """
    RVV 1.0 and 0.7.1 base opd3 implementation with methods shared by all
    opd3 operations
    """

    supports_np = False

    def __init__(self,
                 asmwrap : Callable[[str],str]):

        self.asmwrap = asmwrap
        self.operand_order = [2,0,1]

        self.signatures = make_rvv_opd3_signatures(supports_np=self.supports_np)

    def get_signatures(self) -> list[operation_signature]:
        return self.signatures

    @abstractmethod
    def get_base_inst(self, modifiers : set[mod]) -> str:
        """
        Return the base instruction name based on the specified modifiers

        :param modifiers: set of modifiers to check the name for
        :type modifiers: set[class:`asmgen.asmblocks.operations.opd3_modifier`]
        :return: ASM instruction name
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)


    def diagnose_failure(self, modifiers : set[mod],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):
        if mod.REGIDX in modifiers:
            raise ValueError("RVV has no regidx form")
        if mod.IDX in modifiers:
            raise ValueError("RVV has no idx form")
        if mod.PART in modifiers:
            raise ValueError("RVV has no partial instructions (using vgroups instead)")
        if mod.MASK in modifiers:
            raise NotImplementedError("RVV masked opd3 not implemented yet")


    def inst_prefix(self, a_dt : adt, b_dt : adt, c_dt : adt) -> str:
        """
        Return the first characters of the required instruction
        depending on the data type

        :param a_dt: Data type of the A component
        :type a_dt: class:`asmgen.registers.asm_data_type`
        :param b_dt: Data type of the B component
        :type b_dt: class:`asmgen.registers.asm_data_type`
        :param c_dt: Data type of the C component
        :type c_dt: class:`asmgen.registers.asm_data_type`
        :return: "vf" for FP types, "v" for INT types
        :rtype: str
        """
        _ = (b_dt, c_dt) # explicitly unused, possibly never relevant
        if adt_is_float(a_dt):
            return "vf"
        if adt_is_int(a_dt):
            return "v"

        raise RuntimeError("Unsupported datatype")

    def inst_suffix(self, a_dt : adt, b_dt : adt, c_dt : adt) -> str:
        """
        Return the last characters of the required instruction
        depending on the data type

        :param a_dt: Data type of the A component
        :type a_dt: class:`asmgen.registers.asm_data_type`
        :param b_dt: Data type of the B component
        :type b_dt: class:`asmgen.registers.asm_data_type`
        :param c_dt: Data type of the C component
        :type c_dt: class:`asmgen.registers.asm_data_type`
        :return: "" for FP and signed INT types, "u" for unsigned INT types
            "su" for signed A and unsigned B, "us" for unsigned A and signed B
        :rtype: str
        """
        if adt_is_float(a_dt) and adt_is_float(b_dt) and adt_is_float(c_dt):
            return ""
        if adt_is_signed(a_dt) and adt_is_signed(b_dt) and adt_is_signed(c_dt):
            return ""
        if adt_is_unsigned(a_dt) and adt_is_unsigned(b_dt) and adt_is_unsigned(c_dt):
            return "u"
        if adt_is_signed(a_dt) and adt_is_unsigned(b_dt):
            return "su"
        if adt_is_unsigned(a_dt) and adt_is_signed(b_dt):
            return "us"

        raise RuntimeError("Unsupported datatype")


    # modfier set is only read, therefore a mutable default is ok
    # pylint: disable-next=dangerous-default-value,too-many-locals
    def implementation(self, *,
                       adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                       a_dt : adt, b_dt : adt, c_dt : adt,
                       modifiers : set[mod] = None,
                       **kwargs) -> str:

        modifiers = modifiers or set()


        pref = self.inst_prefix(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)
        mix_pref = "w" if adt_size(c_dt)>adt_size(a_dt) else ""
        suf = self.inst_suffix(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)

        if mod.VF in modifiers and adt_is_int(c_dt):
            form_suf = "vx"
        elif mod.VF in modifiers and adt_is_float(c_dt):
            form_suf = "vf"
        else:
            form_suf = "vv"

        base_inst = self.get_base_inst(modifiers=modifiers)

        inst = pref+mix_pref+base_inst+suf+"."+form_suf

        operands=[f"{adreg}",f"{bdreg}",f"{cdreg}"]

        operands_string = ','.join([operands[i] for i in self.operand_order])

        inst_str = f"{inst} {operands_string}"

        return self.asmwrap(inst_str)
