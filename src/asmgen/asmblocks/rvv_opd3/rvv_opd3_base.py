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
from ..operations import opd3,widening_method,opd3_modifier as mod
from ...util import NIE_MESSAGE

from ..types.riscv64_types import riscv64_freg
from ..types.rvv_types import rvv_vreg

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
    def get_base_inst(self, modifiers : set[mod]) -> str:
        """
        Return the base instruction name based on the specified modifiers

        :param modifiers: set of modifiers to check the name for
        :type modifiers: set[class:`asmgen.asmblocks.operations.opd3_modifier`]
        :return: ASM instruction name
        :rtype: str
        """
        raise NotImplementedError(NIE_MESSAGE)

    @property
    def widening_method(self) -> widening_method:
        return widening_method.VEC_GROUP


    def check_modifiers(self, modifiers : set[mod]):
        if mod.REGIDX in modifiers:
            raise ValueError("RVV has no regidx form")
        if mod.IDX in modifiers:
            raise ValueError("RVV has no idx form")
        if mod.PART in modifiers:
            raise ValueError("RVV has no partial instructions (using vgroups instead)")
        if mod.MASK in modifiers:
            raise NotImplementedError("RVV masked opd3 not implemented yet")

    def check_triple_and_modifiers(self,
                                   a_dt : adt, b_dt : adt, c_dt : adt,
                                   modifiers : set[mod]):
        """
        Combined datatype triple and modifier check


        :param a_dt : Data type of the A component
        :type a_dt : class:`asmgen.registers.asm_data_type`
        :param b_dt : Data type of the B component
        :type b_dt : class:`asmgen.registers.asm_data_type`
        :param c_dt : Data type of the C component
        :type c_dt : class:`asmgen.registers.asm_data_type`
        :param modifiers: set containing the modifiers to check
        :type modifiers: set[class:`asmgen.asmblocks.operations.opd3_modifier`]
        :raises ValueError: If an unsupported modifier/datatype is in the specified set
        """


    def supported_dts(self) -> list[dict[str,adt]]:
        return [
            {'adreg':adt.FP64, 'bdreg':adt.FP64, 'cdreg':adt.FP64},
            {'adreg':adt.FP32, 'bdreg':adt.FP32, 'cdreg':adt.FP32},
            {'adreg':adt.FP16, 'bdreg':adt.FP16, 'cdreg':adt.FP16},

            {'adreg':adt.FP32, 'bdreg':adt.FP32, 'cdreg':adt.FP64},
            {'adreg':adt.FP16, 'bdreg':adt.FP16, 'cdreg':adt.FP32},

            {'adreg':adt.SINT64, 'bdreg':adt.SINT64, 'cdreg':adt.SINT64},
            {'adreg':adt.SINT32, 'bdreg':adt.SINT32, 'cdreg':adt.SINT32},
            {'adreg':adt.SINT16, 'bdreg':adt.SINT16, 'cdreg':adt.SINT16},
            {'adreg':adt.SINT8, 'bdreg':adt.SINT8, 'cdreg':adt.SINT8},

            {'adreg':adt.SINT32, 'bdreg':adt.SINT32, 'cdreg':adt.SINT64},
            {'adreg':adt.SINT16, 'bdreg':adt.SINT16, 'cdreg':adt.SINT32},
            {'adreg':adt.SINT8, 'bdreg':adt.SINT8, 'cdreg':adt.SINT16},

            {'adreg':adt.UINT32, 'bdreg':adt.UINT32, 'cdreg':adt.UINT64},
            {'adreg':adt.UINT16, 'bdreg':adt.UINT16, 'cdreg':adt.UINT32},
            {'adreg':adt.UINT8, 'bdreg':adt.UINT8, 'cdreg':adt.UINT16},

            {'adreg':adt.SINT32, 'bdreg':adt.UINT32, 'cdreg':adt.SINT64},
            {'adreg':adt.SINT16, 'bdreg':adt.UINT16, 'cdreg':adt.SINT32},
            {'adreg':adt.SINT8, 'bdreg':adt.UINT8, 'cdreg':adt.SINT16},

            {'adreg':adt.UINT32, 'bdreg':adt.SINT32, 'cdreg':adt.SINT64},
            {'adreg':adt.UINT16, 'bdreg':adt.SINT16, 'cdreg':adt.SINT32},
            {'adreg':adt.UINT8, 'bdreg':adt.SINT8, 'cdreg':adt.SINT16},
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
    def implementation(self, *,
                       adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                       a_dt : adt, b_dt : adt, c_dt : adt,
                       modifiers : set[mod] = set(),
                       **kwargs) -> str:


        invalid_regs = False
        # check registers
        if mod.VF not in modifiers:
            if not all(isinstance(r, rvv_vreg) for r in (adreg,bdreg,cdreg)):
                invalid_regs = True
        else:
            if not all(isinstance(r, rvv_vreg) for r in (adreg,cdreg)) or \
                    not isinstance(bdreg, riscv64_freg):
                invalid_regs = True

        if invalid_regs:
            raise ValueError(
                    ("Either all dregs of an RVV opd3 must be rvv_vreg"
                     " or a and c must be rvv_vreg and b must be riscv64_freg"))

        # RVV specific check not covered by standard checks
        if adt_size(a_dt) < adt_size(c_dt):
            if adt_is_int(c_dt) and mod.NP in modifiers:
                raise ValueError(
                        "RVV has no np form for widening integer operation")

        pref = self.inst_prefix(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)
        mix_pref = "w" if adt_size(c_dt)>adt_size(a_dt) else ""
        suf = self.inst_suffix(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)
        form_suf = "vf" if mod.VF in modifiers else "vv"

        base_inst = self.get_base_inst(modifiers=modifiers)

        inst = pref+mix_pref+base_inst+suf+"."+form_suf

        operands=[f"{adreg}",f"{bdreg}",f"{cdreg}"]

        operands_string = ','.join([operands[i] for i in self.operand_order])

        inst_str = f"{inst} {operands_string}"

        return self.asmwrap(inst_str)
