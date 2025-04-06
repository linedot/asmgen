"""
SVE fmul instruction
"""
from ...registers import data_reg,asm_data_type as adt
from ..neon_opd3.neon_fmul import neon_fmul
from ..operations import modifier

class sve_fmul(neon_fmul):
    """
    SVE implementation of fmul
    """
    # modifier set is only read, therefore a mutable default is ok
    # pylint: disable-next=dangerous-default-value,too-many-locals,too-many-branches
    def __call__(self, *, adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                 a_dt : adt, b_dt : adt, c_dt : adt,
                 modifiers : set[modifier] = set(),
                 **kwargs) -> str:

        sve_preg = 'p0/m'
        if modifier.MASK in modifiers:
            if 'mreg' not in kwargs:
                raise ValueError("MASK modifier, but no mreg parameter passed")
            sve_preg=kwargs['mreg']+"/m"
        return super().__call__(adreg=adreg, bdreg=bdreg, cdreg=cdreg,
                         a_dt=a_dt, b_dt=b_dt, c_dt=c_dt,
                         modifiers=modifiers,sve_preg=sve_preg,**kwargs)
