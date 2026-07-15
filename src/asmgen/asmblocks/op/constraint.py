# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Classes for operand restrictions/constraints
"""

from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass

from ...registers import (
    data_reg,
    greg_base,
    mreg_base,
    asm_data_type as adt
)

class ArgumentDependencyError(Exception):
    def __init__(self, name: str, deps: list[str]):
        self.name = name
        self.deps = deps
        depstr = ", ".join(deps)
        self.msg = f"unset dependency(ies) for {name}: {depstr}"

        super().__init__(self.msg)

class ConstraintDoesNotApplyError(Exception):
    def __init__(self):
        super().__init__("This constraint does not apply in this specific case")

type value_type = data_reg|greg_base|mreg_base|adt|int


class operand_constraint(ABC):

    def __init__(self):
        self.params = {}

    def specialize_params(name : str,
                          modifiers : set[Enum],
                          context : dict[str,value_type],
                          params : dict[str,value_type]):
        """
        modifies context based on specified modifiers and returns it,
        leaving original context unchanged. 
        To be overriden by inheriting class.

        :param name: name of the argument for which to specialize params
        :param modifiers: modifiers to apply to the operation 
                          (like opd3_modifier.* or opdna1_modifier.*)
        :param context: arguments passed to the operation or already
                        assigned arguments when generating valid values
        :param params: constraint-relevant parameters

        :raises ValueError: if the value is invalid or 
        :raises ConstraintDoesNotApplyError: The constraint does not apply
            (Example: constraint only applies when a dreg is a vreg)
        """


    def __call__(self, name : str,
                 modifiers : set[Enum],
                 val : value_type,
                 context: dict[str,value_type]):
        """
        Use this to validate the value for a given operand
        
        :param name: name of the argument (like 'adreg')
        :param modifiers: modifiers to apply to the operation 
                          (like opd3_modifier.* or opdna1_modifier.*)
        :param val: value to check for the argument
        :param context: dictionary of already assigned argument values

        :raises ArgumentDependencyError: If the operand depends on the value
            of another operand and it is unset
        :raises ValueError: if the value is invalid
        :raises ConstraintDoesNotApplyError: The constraint does not apply
            (Example: constraint only applies when a dreg is a vreg)
        """

        params = {}
        if modifiers:
            if type(modifiers[0]) == Enum:
                raise ValueError("Modfier type can't be a raw Enum")
            if any(type(m) != type(modifiers[0]) for m in modifiers):
                raise ValueError("All modifiers must be of the same type")

            self.specialize_params(
                    name=name,
                    modifiers=modifiers,
                    context=context, params=self.params)

        self.validate(name=name, val=val, context=context, params=params)

    def __iter__(self, name : str,
                 modifiers: set[Enum],
                 context: dict[str,value_type]) \
                         -> Iterable[value_type]:

        """
        Use this to iterate over valid values for a given operand
        
        :param name: name of the argument (like 'adreg')
        :param modifiers: modifiers to apply to the operation 
                          (like opd3_modifier.* or opdna1_modifier.*)
        :param context: dictionary of already assigned argument values

        :raises ArgumentDependencyError: If the operand depends on the value
                                         of another operand and it is unset
        :raises ConstraintDoesNotApplyError: The constraint does not apply
            (Example: constraint only applies when a dreg is a vreg)
        """
        params = {}
        if modifiers:
            if type(modifiers[0]) == Enum:
                raise ValueError("Modfier type can't be a raw Enum")
            if any(type(m) != type(modifiers[0]) for m in modifiers):
                raise ValueError("All modifiers must be of the same type")
            self.specialize_params(
                    name=name,
                    modifiers=modifiers,
                    context=context, params=self.params)

        for v in self.valid_values(name=name, context=context, params=params):
            yield v

    @abstractmethod
    def validate(self,
                 name : str,
                 val : value_type,
                 context : dict[str,value_type],
                 params : dict[str,value_type]):
        """
        Checks if a value is valid for an argument and raises an Error if it is not.
        To be implemented by an inheriting class
        
        :param name: name of the argument (like 'adreg')
        :param context: other argument values
        :param params: Constraint parameters

        :raises ArgumentDependencyError: If the operand depends on the value
                                         of another operand and it is unset
        :raises ValueError: if the value is invalid
        """

    @abstractmethod
    def valid_values(self,
                     name : str,
                     context : dict[str,value_type],
                     params : dict[str, value_type]):
        """
        Returns an iterable over valid values for a given operand. To
        be implemented by an inheriting class
        
        :param name: name of the argument (like 'adreg')
        :param context: other argument values
        :param params: Constraint parameters

        :raises ArgumentDependencyError: If the operand depends on the value
                                         of another operand and it is unset
        """

@dataclass(kw_only=True)
class intval_constraint(operand_constraint):
    """

    """
    what : str = "value"
    getint : Callable[[value_type],int] = lambda v : v
    makeval : Callable[[int], value_type] = lambda d : d

@dataclass(kw_only=True)
class minmax_constraint(intval_constraint):
    minval: int
    maxval: int
    def __post_init__(self):
        self.params['minval'] = self.minval
        self.params['maxval'] = self.maxval

    def validate(self, name : str,
                 val : value_type,
                 context : dict[str,value_type],
                 params : dict[str,value_type]):

        minval = params['minval']
        maxval = params['maxval']

        if self.getint(val) < minval:
            raise ValueError(f"{self.what} of {self.name} must be >= {minval}")
        if self.getint(val) > maxval:
            raise ValueError(f"{self.what} of {self.name} must be <= {maxnval}")

    def valid_values(self, name : str, context : dict[str, value_type]):

        minval = params['minval']
        maxval = params['maxval']

        for i in range(minval, maxval+1):
            yield self.makeval(i)

@dataclass(kw_only=True)
class oneof_constraint(intval_constraint):
    valset: set[int]
    def __post_init__(self):
        self.params['valset'] = self.valset

    def validate(self, name : str,
                 val : value_type,
                 context : dict[str,value_type],
                 params : dict[str,value_type]):

        valset = params['valset']

        if self.getint(val) not in valset:
            raise ValueError(f"{self.what} of {self.name} must be one of: {valset}")

    def valid_values(self, name : str,
                     context : dict[str, value_type],
                     params : dict[str|value_type]):

        valset = params['valset']
        
        for i in valset:
            yield self.makeval(i)

@dataclass(kw_only=True)
class multiple_constraint(intval_constraint):
    multiple: int
    def __post_init__(self):
        self.params['multiple'] = self.multiple

    def validate(self, name : str, val : value_type, context : dict[str,value_type], params : dict[str,value_type]):

        multiple = params['multiple']

        if 0 != (self.getval(val) % multiple):
            raise ValueError(f"{self.what} of {self.name} must be a multiple of {multiple}")

    def valid_values(self, name : str, context : dict[str, value_type]):

        multiple = params['multiple']
        minval = params.get('minval', 0)
        maxval = params.get('maxval', 32)

        for i in range(minval, maxval+1, multiple):
            yield self.makeval(i)


@dataclass(kw_only=True)
class otherplusn_constraint(intval_constraint):
    other: str
    offset: int
    def __post_init__(self):
        self.params['other'] = self.other
        self.params['offset'] = self.offset

    def validate(self, name : str,
                 val : value_type,
                 context : dict[str,value_type],
                 params : dict[str,value_type]):

        other = params['other']
        otherint = self.getint(context[other])
        offset = params['offset']

        if self.getint(val) != otherint+offset:
            raise ValueError((f"{self.what} of {self.name} must be "
                              f"{self.what} of {other} plus {offset}"))

    def valid_values(self, name : str, context : dict[str, value_type]):

        other = params['other']
        otherint = self.getint(context[other])
        offset = params['offset']

        yield self.makeval(otherint+offset)


@dataclass(kw_only=True)
class otherplusnmod_constraint(intval_constraint):
    other: str
    offset: int
    modval: int
    def __post_init__(self):
        self.params['other'] = self.other
        self.params['offset'] = self.offset
        self.params['modval'] = self.modval

    def validate(self, name : str,
                 val : value_type,
                 context : dict[str,value_type],
                 params : dict[str,value_type]):

        other = params['other']

        if other not in context:
            raise ArgumentDependencyError(name=name, deps=[other])

        otherint = self.getint(context[other])
        offset = params['offset']
        modval = params['modval']

        if self.getint(val) != (otherint+offset) % modval:
            raise ValueError((f"{self.what} of {self.name} must be "
                              f"{self.what} of {other} plus {offset} "
                              f"modulo {modval}"))

    def valid_values(self, name : str, context : dict[str, value_type]):

        other = params['other']
        otherint = self.getint(context[other])
        offset = params['offset']
        modval = params['modval']

        yield self.makeval((otherint+offset) % modval)

