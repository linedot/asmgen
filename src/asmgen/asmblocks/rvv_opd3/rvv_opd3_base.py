# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RVV 1.0 and 0.7.1 opd3 base
"""
from abc import abstractmethod
from typing import Callable

from ...registers import (
    asm_data_type as adt,
    adt_triple,
    adt_size,
    adt_is_float,adt_is_int,
    adt_is_signed,adt_is_unsigned,
    data_reg
)
from ..operations import opd3,widening_method,modifier

from ...util import NIE_MESSAGE

class rvv_opd3_base(opd3):
    """
    RVV 1.0 and 0.7.1 base opd3 implementation with methods shared by all
    opd3 operations
    """

    def __init__(self,
                 asmwrap : Callable[[str],str]):
        self.asmwrap = asmwrap

        self.operand_order = [2,0,1]

    @abstractmethod
    def get_base_inst(self, modifiers : set[modifier]) -> str:
        """
        Return the base instruction name based on the specified modifiers

        :param modifiers: set of modifiers to check the name for
        :type modifiers: set[class:`asmgen.asmblocks.operations.modifier`]
        :return: ASM instruction name
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @property
    def widening_method(self) -> widening_method:
        return widening_method.VEC_GROUP

    def check_modifiers(self, modifiers : set[modifier]):
        if modifier.REGIDX in modifiers:
            raise ValueError("RVV has no regidx form")
        if modifier.IDX in modifiers:
            raise ValueError("RVV has no idx form")
        if modifier.PART in modifiers:
            raise ValueError("RVV has no partial instructions (using vgroups instead)")

    def check_triple_and_modifiers(self,
                                   a_dt : adt, b_dt : adt, c_dt : adt,
                                   modifiers : set[modifier]):
        """
        Combined datatype triple and modifier check


        :param a_dt : Data type of the A component
        :type a_dt : class:`asmgen.registers.asm_data_type`
        :param b_dt : Data type of the B component
        :type b_dt : class:`asmgen.registers.asm_data_type`
        :param c_dt : Data type of the C component
        :type c_dt : class:`asmgen.registers.asm_data_type`
        :param modifiers: set containing the modifiers to check
        :type modifiers: set[class:`asmgen.asmblocks.operations.modifier`]
        :raises ValueError: If an unsupported modifier/datatype is in the specified set
        """
        super().check_triple(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)
        self.check_modifiers(modifiers=modifiers)

        if a_dt != b_dt:
            raise ValueError("A and B must have same type")
        if adt_size(a_dt) > adt_size(c_dt):
            raise ValueError("C type can't have smaller size than A/B type")
        if (adt_is_float(c_dt) and adt_is_int(a_dt)) or\
           (adt_is_float(a_dt) and adt_is_int(c_dt)):
            raise ValueError("Accumulator and multiplicands must be both either fp or int types")
        if adt_size(a_dt) < adt_size(c_dt):
            if adt_is_int(c_dt) and modifier.NP in modifiers:
                raise ValueError("RVV has no np form for widening integer operation")
        if (adt_size(c_dt)//adt_size(a_dt)) > 2:
            raise ValueError("RVV only supports 2*SEW widening")

    def supported_triples(self) -> list[adt_triple]:
        return [
            adt_triple(a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64),
            adt_triple(a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32),
            adt_triple(a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16),

            adt_triple(a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP64),
            adt_triple(a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP32),

            adt_triple(a_dt=adt.SINT64, b_dt=adt.SINT64, c_dt=adt.SINT64),
            adt_triple(a_dt=adt.SINT32, b_dt=adt.SINT32, c_dt=adt.SINT32),
            adt_triple(a_dt=adt.SINT16, b_dt=adt.SINT16, c_dt=adt.SINT16),
            adt_triple(a_dt=adt.SINT8,  b_dt=adt.SINT8,  c_dt=adt.SINT8),

            adt_triple(a_dt=adt.SINT32, b_dt=adt.SINT32, c_dt=adt.SINT64),
            adt_triple(a_dt=adt.SINT16, b_dt=adt.SINT16, c_dt=adt.SINT32),
            adt_triple(a_dt=adt.SINT8,  b_dt=adt.SINT8,  c_dt=adt.SINT16),

            adt_triple(a_dt=adt.UINT32, b_dt=adt.UINT32, c_dt=adt.UINT64),
            adt_triple(a_dt=adt.UINT16, b_dt=adt.UINT16, c_dt=adt.UINT32),
            adt_triple(a_dt=adt.UINT8,  b_dt=adt.UINT8,  c_dt=adt.UINT16),

            adt_triple(a_dt=adt.SINT32, b_dt=adt.UINT32, c_dt=adt.SINT64),
            adt_triple(a_dt=adt.SINT16, b_dt=adt.UINT16, c_dt=adt.SINT32),
            adt_triple(a_dt=adt.SINT8,  b_dt=adt.UINT8,  c_dt=adt.SINT16),

            adt_triple(a_dt=adt.UINT32, b_dt=adt.SINT32, c_dt=adt.SINT64),
            adt_triple(a_dt=adt.UINT16, b_dt=adt.SINT16, c_dt=adt.SINT32),
            adt_triple(a_dt=adt.UINT8,  b_dt=adt.SINT8,  c_dt=adt.SINT16),
        ]

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
    def __call__(self, *, adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                 a_dt : adt, b_dt : adt, c_dt : adt,
                 modifiers : set[modifier] = set(),
                 **kwargs) -> str:

        self.check_triple_and_modifiers(
                a_dt=a_dt, b_dt=b_dt, c_dt=c_dt,
                modifiers=modifiers)


        pref = self.inst_prefix(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)
        mix_pref = "w" if adt_size(c_dt)>adt_size(a_dt) else ""
        suf = self.inst_suffix(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)
        form_suf = "vf" if modifier.VF in modifiers else "vv"

        base_inst = self.get_base_inst(modifiers=modifiers)

        inst = pref+mix_pref+base_inst+suf+"."+form_suf

        operands=[f"{adreg}",f"{bdreg}",f"{cdreg}"]

        operands_string = ','.join([operands[i] for i in self.operand_order])

        inst_str = f"{inst} {operands_string}"

        return self.asmwrap(inst_str)
