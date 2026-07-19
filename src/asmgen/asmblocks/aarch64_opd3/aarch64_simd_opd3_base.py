# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
AArch64 SIMD opd3 base class for both NEON and SVE
"""

from typing import Callable,Any

from ...registers import (
    asm_data_type as adt,
    adt_size,
    adt_is_unsigned,
    data_reg
)
from ..op import (
    opd3,
    opd3_modifier as mod,
    operation_signature
)

class aarch64_simd_opd3_base(opd3):
    """
    NEON+SVE base opd3 implementation with methods shared by all
    opd3 operations
    """
    inst_base = "invalid"
    has_acc_suffix = False

    def __init__(self,
                 asmwrap : Callable[[str],str],
                 dt_suffixes : dict[adt,str],
                 dt_idxsuffixes : dict[adt,str]
                 ):
        self.asmwrap = asmwrap
        self.dt_suffixes = dt_suffixes
        self.dt_idxsuffixes = dt_idxsuffixes

        self.signatures = []


    def get_signatures(self) -> list[operation_signature]:
        return self.signatures

    def diagnose_failure(self,
                         modifiers : set[mod],
                         kwargs : dict[str,Any],
                         dts : dict[str,adt]):
        if mod.VF in modifiers:
            raise ValueError("NEON/SVE has no vf form")
        if mod.REGIDX in modifiers:
            raise ValueError("NEON/SVE has no regidx form")

        if mod.IDX in modifiers and 'idx' not in kwargs:
            raise ValueError("Operand missing: idx")
        if mod.PART in modifiers and 'part' not in kwargs:
            raise ValueError("Operand missing: part")

        a_dt = dts.get('adreg')
        c_dt = dts.get('cdreg')
        if a_dt and c_dt and adt_is_unsigned(a_dt):
            if adt_size(a_dt) == adt_size(c_dt):
                op_name = self.inst_base.upper()
                raise ValueError(
                    f"Only widening unsigned integer {op_name} supported in NEON"
                )


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


    def implementation(self, *,
                       adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                       a_dt : adt, b_dt : adt, c_dt : adt,
                       modifiers : set[mod] = None,
                       **kwargs) -> str:

        if modifiers is None:
            modifiers = set()

        # This allows the SVE version to use the same codepath
        predicate = f"{kwargs['amreg']}/m," if 'amreg' in kwargs else ""

        # Add a/s suffix if instruction supports it
        suf = ""
        if self.has_acc_suffix:
            suf = "s" if mod.NP in modifiers else "a"

        inst = self.inst_prefix(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt) +\
               self.inst_base + suf

        if mod.PART in modifiers:
            inst += self.partial_inst_suffix(
                    ways=adt_size(c_dt)//adt_size(a_dt),
                    part=kwargs['part'])

        if {mod.IDX, mod.BLOCKIDX} & modifiers:
            b = f"{bdreg}.{self.dt_idxsuffixes[b_dt]}[{kwargs.get('idx', 0)}]"
        else:
            b = f"{bdreg}.{self.dt_suffixes[b_dt]}"

        return self.asmwrap(
            f"{inst} "
            f"{cdreg}.{self.dt_suffixes[c_dt]},"
            f"{predicate}"
            f"{adreg}.{self.dt_suffixes[a_dt]},"
            f"{b}")
