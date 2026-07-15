from dataclasses import dataclass, field

from ..constraint import (
    operand_constraint,
    minmax_constraint,
    oneof_constraint,
    otherplusnmod_constraint,
    multiple_constraint,
    value_type,
    ConstraintDoesNotApplyError
)

from ..sve.constraints import sve_vreg_index_constraint

from ...types.aarch64_types import aarch64_greg
from ...types.neon_types import neon_vreg


@dataclass(kw_only=True)
class neon_vreg_index_constraint(intval_constraint):
    """
    Constraint on register indices for NEON vregs
    """
    what: str ='index'
    getint: Callable[[value_type],int] = lambda v : v.int
    makeval: Callable[[int],value_type] = lambda i : neon_vreg(reg_idx=i)

@dataclass(kw_only=True)
class neon_vreg_noerror_constraint(neon_vreg_index_constraint):
    """
    Constraint for generating NEON vregs
    """
    def validate(self, name, val, context, params):
        pass

    def valid_values(self, name, context, params):
        for i in range(32):
            yield sve_vreg(reg_idx=i)

@dataclass(kw_only=True)
class neon_otherplusnmod_constraint(otherplusnmod_constraint,
                                    neon_vreg_index_constraint):
    def specialize_params(self, name : str,
                          modifiers : set[Enum],
                          context : dict[str,value_type],
                          params : dict[str,value_type]):
        dreg_count = len(context['dregs'])
        params['modval'] = 32
        params['offset'] = 1

    def validate(self, name : str, val : value_type,
                 context : dict[str,value_type],
                 params : dict[str,value_type]):
        if not isinstance(val, neon_vreg):
            raise ConstraintDoesNotApplyError()

        super().validate(name,val,context,params)
