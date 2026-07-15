# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Operation signatures
"""

from enum import Enum,auto
from dataclasses import dataclass

from .modifier import operation_modifier
from .constraint import operand_constraint,value_type
from .operand import operand_shape,is_register_type,operand_type

from typing import Any

@dataclass(kw_only=True)
class operation_signature:
    """
    Defines physical hardware variant of an instruction
    """
    
    modifiers : set[operation_modifier]
    structural_params : dict[str, Any]
    operands: dict[str, operand_shape]


    def get_tag(self) -> str:
        """
        Tag for this signature
        """
    
        mod_str = "_".join(sorted(m.name for m in self.modifiers))
        param_str = "_".join(f"{k}{v}" for k,v in self.structural_params.items())
        return f"{mod_str}_{param_str}".strip("_")

    def match_intent(self,
                     modifiers: set[operation_modifier],
                     kwargs: dict[str, Any],
                     dts: dict[str, 'adt']) -> bool:
        """
        Check if signature fulfills intent

        :param modifiers: operation modifiers
        :param kwargs: all parameters to operation (to check structural params of)
        :param dts: data types of operands
        """

        if self.modifiers != modifiers:
            return False

        for param, val in self.structural_params.items():
            if kwargs.get(param) != val:
                return False

        for name, dt in dts.items():
            if name in self.operands and self.operands[name].dt != dt:
                return False

        return True

    def validate_allocation(self, kwargs: dict[str, Any]):
        """
        Check if valid operands were allocated
        """

        for name, shape in self.operands.items():
            if name not in kwargs:
                raise ValueError(f"Missing operand: {name}")

            val = kwargs[name]

            if shape.otype == operand_type.REGISTER and \
                    not is_register_type(val, shape.rtype):
                raise TypeError(f"{name} must be {shape.rtype.name}, got {type(val)}")

            for constraint in shape.value_constraints:
                constraint.validate(name=name, val=val, context=kwargs)


# Some notes/thoughts:

# What is a "use"/part of the signature?

#fma{NP}(adreg.v.FP64, bdreg.v.FP64, cdreg.v.FP64)
#ld{STRUCT}(adreg.v.FP64,bdreg.v.FP64,cdreg.v.FP64,ddreg.v.FP64,agreg.UINT64,nstructs=4,ioffset)
#ld{LANE}(adreg.v.FP32,agreg.UINT64,lane,ioffset)
#ld{GATHER}(adreg.v.FP64,agreg,offvreg.SINT64)
#fmul{VF}(adreg.v.FP16,bdreg.f.FP16,cdreg.v.FP32, widening_method=VEC_MULTI, cdreg2.v.FP32)
#fmul{VF}(adreg.v.FP16,bdreg.f.FP16,cdreg.v.FP32, widening_method=SPLIT_INSTRUCTIONS, part)
#fmul{VF}(adreg.v.FP16,bdreg.f.FP16,cdreg.v.FP32, widening_method=VEC_GROUP)
#st{ROW}(adreg.t.FP32, agreg, voffset, rowreg=agreg.SINT32, immrow)

# possible dependencies
# STRUCT -> nstructs -> len(dregs)
# A=FP16,B=FP16,C=FP32 -> widening_method -> number of cdreg OR which part OR index consraint

# -> data type must be part of signature, since other arguments can depend on it
# only dependency of the actual signature on exact value of an additional parameter for now is "nstructs", but more could be possible?

# What constraints are not part of a signature?
# - constraint on register index
# - constraint on immediate value
