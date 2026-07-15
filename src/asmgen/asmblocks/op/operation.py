# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Abstract base classes for arithmetic operations/instructions
"""

from abc import ABC,abstractmethod
from enum import Enum,auto

from ...registers import (
    data_reg,
    asm_data_type as adt,
    adt_triple,
)
from .constraint import operand_constraint
from .misc import make_ord_prefix

class operation(ABC):
    """
    Abstraction over operations/instructions
    """

    NIE_MESSAGE="Inheriting class must implement this method"

    @abstractmethod
    def get_signatures(self) -> list[operation_signature]:
        """
        Returns a list of operation signatures this operation supports

        :return: List of valid signatures
        """

    def diagnose_failure(self, modifiers: set[Enum], kwargs: dict, dts: dict[str, adt]):
        """
        Optional hook for inheriting classes to provide domain-specific 
        error messages.

        :raises ValueError: if a specific bad pattern is found.
        """
        pass

    def _auto_diagnose(self, modifiers: set[Enum], kwargs: dict, dts: dict[str, adt]):
        """
        Automatically infers basic errors by looking at the pooled signatures.
        """
        sigs = self.get_signatures()
        
        all_supported_mods = set().union(*(sig.modifiers for sig in sigs))
        unsupported_mods = modifiers - all_supported_mods
        if unsupported_mods:
            unsup_mod_string = "|".join(m.name for m in unsupported_mods)
            raise ValueError(f"{type(self).__name__} does not support these modifiers at all: {{{unsup_mod_string}}}")


    def execute(self, *,
                dregs : list[data_reg],
                gregs : list[greg_type],
                dts : dict[str,adt],
                modifiers : set[Enum],
                **kwargs) -> str:
        """
        Performs checks on all arguments, generates the parameters for the underlying
        implementation and calls it
        """

        sigs = self.get_signatures()


        matched_sig = next(
            (s for s in sigs if \
                s.match_intent(modifiers=modifiers,
                               kwargs=kwargs,
                               dts=dts)),
            None)
        if not matched_sig:
            self.diagnose_failure(modifiers, kwargs, dts)
            self._auto_diagnose(modifiers, kwargs, dts)
            
            raise ValueError(
                f"Invalid configuration for {type(self).__name__}. "
                f"Modifiers: {modifiers}, dts: {dts}"
            )


        resolved_operands = kwargs.copy()

        # generate dreg args
        for i,reg in enumerate(dregs):
            pfx = make_ord_prefix(i)
            dreg_name = f"{pfx}dreg"
            dt_name = f"{pfx}_dt"
            resolved_operands[dreg_name] = reg
            resolved_operands[dt_name] = dts[dreg_name]

        for i,reg in enumerate(gregs):
            pfx = make_ord_prefix(i)
            greg_name = f"{pfx}greg"
            resolved_operands[greg_name] = reg

        matched_sig.validate_allocation(resolved_operands)

        resolved_operands['modifiers'] = modifiers
        
        # opdna1 has a different interface
        resolved_operands['dregs'] = dregs

        return self.implementation(**resolved_operands)


    @abstractmethod
    def implementation(self, **kwargs) -> str:
        """
        Actual implementation of the operation to be implemented
        by the inheriting class
        """
        raise NotImplementedError(self.NIE_MESSAGE)


