from dataclasses import dataclass

from ..constraint import (
    operand_constraint,
    minmax_constraint,
    oneof_constraint,
    otherplusnmod_constraint,
    multiple_constraint,
    intval_constraint,
    value_type,
    ConstraintDoesNotApplyError
)

from ...types.aarch64_types import aarch64_greg
from ...types.sve_types import sve_vreg

@dataclass(kw_only=True)
class sve_vreg_index_constraint(intval_constraint):
    """
    Constraint on register indices for SVE vregs
    """
    what: str ='index'
    getint: Callable[[value_type],int] = lambda v : v.int
    makeval: Callable[[int],value_type] = lambda i : sve_vreg(reg_idx=i)

@dataclass(kw_only=True)
class sve_vreg_noerror_constraint(sve_vreg_index_constraint):
    """
    Constraint for generating vregs
    """
    def validate(self, name, val, context, params):
        pass

    def valid_values(self, name, context, params):
        # only 1 za register possible
        if 'adreg' != name:
            raise ConstraintDoesNotApplyError()
        
        for i in range(32):
            yield sve_vreg(reg_idx=i)
