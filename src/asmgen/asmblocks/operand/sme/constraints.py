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
from ...types.sve_types import sve_vreg
from ...types.sme_types import sme_treg

@dataclass(kw_only=True)
class nt_dreg_otherplusnmod_constraint(otherplusnmod_constraint,
                                       sve_vreg_index_constraint):
    def specialize_params(self, name : str,
                          modifiers : set[Enum],
                          context : dict[str,value_type],
                          params : dict[str,value_type]):
        dreg_count = len(context['dregs'])
        params['modval'] = 32
        params['offset'] = 1
        if 2 == dreg_count:
            params['offset'] = 8
        if 4 == dreg_count:
            params['offset'] = 4

    def validate(self, name : str, val : value_type,
                 context : dict[str,value_type],
                 params : dict[str,value_type]):
        if not isinstance(val, sve_vreg):
            raise ConstraintDoesNotApplyError()

        super().validate(name,val,context,params)

@dataclass(kw_only=True)
class nt_dreg_oneof_constraint(oneof_constraint,
                               sve_vreg_index_constraint):
    def specialize_params(self, name : str,
                          modifiers : set[Enum],
                          context : dict[str,value_type],
                          params : dict[str,value_type]):

        if 'adreg' != name:
            raise ConstraintDoesNotApplyError()

        dreg_count = len(context['dregs'])
        if dreg_count == 2:
            params['valset'] = {*range(8), *range(16, 24)}
        elif dreg_count == 4:
            params['valset'] = {*range(4), *range(16, 20)}
        else:
            params['valset'] = set(range(32))

    def validate(self, name : str, val : value_type,
                 context : dict[str,value_type],
                 params : dict[str,value_type]):
        if not isinstance(val, sve_vreg):
            raise ConstraintDoesNotApplyError()

        super().validate(name,val,context,params)


@dataclass(kw_only=True)
class sme_treg_index_constraint(intval_constraint):
    """
    Constraint on register indices for SME tregs
    """
    what: str ='index'
    getint: Callable[[value_type],int] = lambda v : v.int
    makeval: Callable[[int],value_type] = lambda i : sme_treg(reg_idx=i)

@dataclass(kw_only=True)
class sme_treg_noerror_constraint(sme_treg_index_constraint):
    """
    Constraint for generating tregs
    """
    def validate(self, name, val, context, params):
        pass

    def valid_values(self, name, context, params):
        # only 1 za register possible
        if 'adreg' != name:
            raise ConstraintDoesNotApplyError()

        dt = context['a_dt']
        
        for i in range(adt_size(dt)):
            yield sme_treg(reg_idx=i, dt=dt)
