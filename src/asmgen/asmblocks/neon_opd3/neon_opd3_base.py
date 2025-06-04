# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
NEON/ASIMD opd3 base case
"""

from typing import Callable

from ...registers import (
    asm_data_type as adt,
    adt_size,
    adt_is_float,adt_is_int,
    data_reg
)
from ..operations import opd3,widening_method,modifier

class neon_opd3_base(opd3):
    """
    NEON/ASIMD base opd3 implementation with methods shared by all
    opd3 operations
    """

    inst_base = "invalid"

    def __init__(self,
                 asmwrap : Callable[[str],str],
                 dt_suffixes : dict[adt,str],
                 dt_idxsuffixes : dict[adt,str],
                 ):
        self.asmwrap = asmwrap
        self.dt_suffixes = dt_suffixes
        self.dt_idxsuffixes = dt_idxsuffixes

    @property
    def widening_method(self) -> widening_method:
        return widening_method.SPLIT_INSTRUCTIONS

    def check_modifiers(self, modifiers : set[modifier]):
        if modifier.VF in modifiers:
            raise ValueError("NEON has no vf form")
        if modifier.REGIDX in modifiers:
            raise ValueError("NEON has no regidx form")

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
        valid_c_types = [adt.FP64, adt.FP32, adt.FP16,
                         adt.UINT64, adt.UINT32, adt.UINT16,
                         adt.SINT64, adt.SINT32, adt.SINT16]
        if c_dt not in valid_c_types:
            valid_str = ','.join([str(t) for t in valid_c_types])
            raise ValueError(f"C type must be one of [{valid_str}]")
        if c_dt in [adt.UINT64, adt.UINT32, adt.UINT16]:
            if adt_size(a_dt) == adt_size(c_dt):
                raise ValueError("only widening variants exist for unsigned integer types")


    def inst_prefix(self, a_dt : adt, b_dt : adt, c_dt : adt) -> str:
        """
        Choose the correct instruction prefix based on specified types

        :param a_dt: Type of the A component
        :type a_dt: class:`asmgen.registers.asm_data_type`
        :param b_dt: Type of the B component
        :type b_dt: class:`asmgen.registers.asm_data_type`
        :param c_dt: Type of the C component
        :type c_dt: class:`asmgen.registers.asm_data_type`
        :return: string containing the instruction prefix
        :rtype: str
        """
        if a_dt in [adt.FP8E5M2, adt.FP8E4M3, adt.FP16, adt.FP32, adt.FP64]:
            return "f"
        if a_dt in [adt.BF16]:
            return "bf"
        if a_dt in [adt.SINT8, adt.SINT16, adt.SINT32, adt.SINT64] and (a_dt == c_dt):
            return ""
        if a_dt in [adt.UINT8, adt.UINT16] and b_dt in [adt.UINT8, adt.UINT16]:
            return "u"
        if a_dt in [adt.SINT8, adt.SINT16] and b_dt in [adt.SINT8, adt.SINT16]:
            return "s"

        raise RuntimeError("Unsupported datatype")

    def partial_inst_suffix(self, ways : int, part : int):
        """
        Get the instruction suffix for a n-ways widening instruction computing
        the specified part

        :param ways: widening factor
        :type ways: int
        :param ways: which part to compute
        :type ways: int
        :return: string containing the required suffix
        :rtype: str
        """
        if ways.bit_count() != 1:
            raise ValueError(f"ways={ways} is not a power of 2")
        char_count = ways.bit_length()-1
        suf = 'l'*char_count

        # Works, but is too much of a 'detour' from the intent
        #suf += f"{part:b>{ways.bit_length()-1}b}".replace('1','t').replace('0','b')
        #return suf

        partsuf = ['b']*char_count
        for i in range(char_count):
            if (part >> i) & 0x1:
                partsuf[-(i+1)] = 't'

        suf += "".join(partsuf)

        return suf

    # modfier set is only read, therefore a mutable default is ok
    # pylint: disable-next=dangerous-default-value,too-many-locals,too-many-branches
    def __call__(self, *, adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                 a_dt : adt, b_dt : adt, c_dt : adt,
                 modifiers : set[modifier] = set(),
                 **kwargs) -> str:
        self.check_triple(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)

        part = 0
        if adt_size(a_dt) < adt_size(c_dt):
            if (not modifier.PART in modifiers) or ('part' not in kwargs):
                raise ValueError(
                        "NEON requires 'PART' modifier and argument for widening operations")
            part = kwargs['part']

        # This allows the SVE version to use the same codepath
        sve_preg=""
        if 'sve_preg' in kwargs:
            sve_preg=kwargs['sve_preg'] + ','

        # Add a/s suffix if instruction supports it
        suf = ""
        try:
            self.check_modifiers({modifier.NP})
            suf = "s" if modifier.NP in modifiers else "a"
        except ValueError:
            pass

        idx = 0
        if modifier.IDX in modifiers:
            if 'idx' not in kwargs:
                raise ValueError("'idx' modifier specified, but not 'idx' parameter")
            idx = kwargs['idx']



        inst = self.inst_prefix(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt) +\
               self.inst_base + suf
        if modifier.PART in modifiers:
            ways = adt_size(c_dt)//adt_size(a_dt)
            inst += self.partial_inst_suffix(ways=ways, part=part)

        narrow_suf = self.dt_suffixes[a_dt]
        wide_suf = self.dt_suffixes[c_dt]
        inst_str = f"{inst} {cdreg}.{wide_suf},{sve_preg}{adreg}.{narrow_suf},"
        if modifier.IDX in modifiers:
            b_suf = self.dt_idxsuffixes[b_dt]
            inst_str += f"{bdreg}.{b_suf}[{idx}]"
        else:
            b_suf = narrow_suf
            inst_str += f"{bdreg}.{b_suf}"

        return self.asmwrap(inst_str)
