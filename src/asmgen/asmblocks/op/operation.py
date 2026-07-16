# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Abstract base classes for arithmetic operations/instructions
"""

from abc import ABC,abstractmethod

from ...registers import (
    data_reg,
    greg_base,
    mreg_base,
    asm_data_type as adt,
)

from .signature import operation_signature
from .misc import make_ord_prefix
from .modifier import operation_modifier as mod

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

    def diagnose_failure(self, modifiers: set[mod], kwargs: dict, dts: dict[str, adt]):
        """
        Optional hook for inheriting classes to provide domain-specific 
        error messages.

        :raises ValueError: if a specific bad pattern is found.
        """

    def _auto_diagnose(self, modifiers: set[mod], kwargs: dict, dts: dict[str, adt]):
        """
        Automatically infers basic errors by looking at the pooled signatures.
        """

        # Not using these for auto diagnosis right now
        del kwargs, dts

        sigs = self.get_signatures()

        all_supported_mods = set().union(*(sig.modifiers for sig in sigs))
        unsupported_mods = modifiers - all_supported_mods
        if unsupported_mods:
            unsup_mod_string = "|".join(m.name for m in unsupported_mods)
            raise ValueError(
                    (f"{type(self).__name__} does not support "
                     f"these modifiers at all: {{{unsup_mod_string}}}"))


    def resolve_dregs(self,
                      dregs : list[data_reg],
                      dts : dict[str,adt]) -> dict[str,data_reg]:
        """
        Helper to create named dreg arguments

        :param dregs: list of dregs
        :return: dictionary mapping argument names to registers
        """

        resolved_operands = {}

        for i,reg in enumerate(dregs):
            pfx = make_ord_prefix(i)
            dreg_name = f"{pfx}dreg"
            dt_name = f"{pfx}_dt"
            resolved_operands[dreg_name] = reg
            resolved_operands[dt_name] = dts[dreg_name]

        return resolved_operands

    def resolve_gregs(self, gregs : list[greg_base]) -> dict[str,greg_base]:
        """
        Helper to create named greg arguments

        :param dregs: list of dregs
        :return: dictionary mapping argument names to registers
        """

        resolved_operands = {}

        for i,reg in enumerate(gregs):
            pfx = make_ord_prefix(i)
            greg_name = f"{pfx}greg"
            resolved_operands[greg_name] = reg

        return resolved_operands

    def execute(self, *,
                dregs : list[data_reg],
                gregs : list[greg_base],
                dts : dict[str,adt],
                modifiers : set[mod],
                **kwargs) -> str:
        """
        Performs checks on all arguments, generates the parameters for the underlying
        implementation and calls it

        :param dregs: Data registers to use
        :param gregs: GP registers to use
        :param dts: Data type to use for each named operand
        :param modifiers: operation modifiers
        """


        resolved_operands = kwargs.copy()

        resolved_operands |= self.resolve_dregs(dregs, dts)
        resolved_operands |= self.resolve_gregs(gregs)


        sigs = self.get_signatures()
        matched_sig = next(
            (s for s in sigs if \
                s.match_intent(modifiers=modifiers,
                               kwargs=resolved_operands,
                               dts=dts)),
            None)
        if not matched_sig:
            self.diagnose_failure(modifiers, kwargs, dts)
            self._auto_diagnose(modifiers, kwargs, dts)

            regtypes = [f"{name}:{type(r).__name__}" for name,r in resolved_operands.items() \
                    if isinstance(r, (data_reg,greg_base,mreg_base))]

            regtypestr = ", ".join(regtypes)

            dtstr = ", ".join(f"{name}:{dt.name}" for name,dt in dts.items())

            raise ValueError(
                f"Invalid configuration for {type(self).__name__}.\n"
                f"  Modifiers: {modifiers}\n"
                f"  dts: {dtstr}\n"
                f"  reg types: {regtypestr}"
            )


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
