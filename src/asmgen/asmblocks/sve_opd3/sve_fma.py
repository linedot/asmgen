# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
SVE fma instruction
"""
from ...registers import data_reg,asm_data_type as adt
from ..neon_opd3.neon_fma import neon_fma
from ..operations import opd3_modifier as mod
from ..types.sve_types import sve_vreg

class sve_fma(neon_fma):
    """
    SVE implementation of fma
    """
    # modfier set is only read, therefore a mutable default is ok
    # pylint: disable-next=dangerous-default-value,too-many-locals,too-many-branches
    
    def check_valid_registers(self, dregs : list[data_reg]) -> bool:
        if not all(isinstance(d, sve_vreg) for d in dregs):
            raise ValueError("All dregs of a SVE opd3 must be sve_vreg")

    def __call__(self, *, adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                 a_dt : adt, b_dt : adt, c_dt : adt,
                 modifiers : set[mod] = set(),
                 **kwargs) -> str:
        self.check_triple_and_modifiers(
                a_dt=a_dt, b_dt=b_dt, c_dt=c_dt,
                modifiers=modifiers)

        sve_preg = 'p0/m'
        if mod.MASK in modifiers:
            if 'mreg' not in kwargs:
                raise ValueError("MASK modifier, but no mreg parameter passed")
            sve_preg=kwargs['mreg']+"/m"
        return super().__call__(adreg=adreg, bdreg=bdreg, cdreg=cdreg,
                         a_dt=a_dt, b_dt=b_dt, c_dt=c_dt,
                         modifiers=modifiers,sve_preg=sve_preg,**kwargs)
