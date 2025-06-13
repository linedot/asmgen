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

from ..registers import (
    data_reg,
    asm_data_type as adt,
    adt_triple,
)

class widening_method(Enum):
    """
    Possible methods ISAs can have for widening instructions
    """
    NONE = auto()
    VEC_GROUP = auto()
    VEC_MULTI = auto()
    DOT_NEIGHBOURS = auto()
    SPLIT_INSTRUCTIONS = auto()

class modifier(Enum):
    """
    Possible modifiers for an instruction/operation
    """
    NP = auto()
    IDX = auto()
    REGIDX = auto()
    PART = auto()
    VF = auto()
    MASK = auto()

class opd3(ABC):
    """
    Assembly/IR instruction with 3 data operands

    Data operands means registers. Exanding to memory (shared/tensor mem in GPUs,
    TCMs, x86 mem operands, etc...) operands is planned
    Examples:
      fma      op1, op2, op3      : op3      <-   op1 * op2      + op3
      fma.np   op1, op2, op3      : op3      <- -(op1 * op2)     + op3
      fma.idx  op1, op2, op3, idx : op3      <-   op1 * op2[idx] + op3
      fmul     op1, op2, op3      : op3      <-   op1 * op2
      opa      op1, op2, op3      : op3      <-   op1 o op2      + op3
      dota     op1, op2, op3      : op3      <-   op1 . op2      + op3
      dota.idx op1, op2, op3      : op3[idx] <-   op1 . op2      + op3[idx]
      mma      op1, op2, op3      : op3      <-   op1 x op2      + op3
    (*: elementwise multiplication)
    (o: outer product)
    (.: dot product)
    (x: matrix product)
    """
    NIE_MESSAGE="Method not implemented"

    @property
    @abstractmethod
    def widening_method(self) -> widening_method:
        """
        Return the method used to deal with widening instructions
        
        :return : widening method
        :rtype : class:`asmgen.asmblocks.operations.widening_method`
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    @abstractmethod
    def supported_triples(self) -> list[adt_triple]:
        """
        Return the list of supported data type combinations
        
        :return : list of supported data type combinations
        :rtype : class:`asmgen.registers.adt_triple`
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    @abstractmethod
    def check_modifiers(self, modifiers : set[modifier]):
        """
        Checks whether the operations supports the specified modifiers


        :param modifiers: set containing the modifiers to check
        :type modifiers: set[class:`asmgen.asmblocks.operations.modifier`]
        :raises ValueError: If an unsupported modifier is in the specified set
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    @abstractmethod
    def __call__(self, *, adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                 a_dt : adt, b_dt : adt, c_dt : adt,
                 modifiers : set[modifier], **kwargs) -> str:
        """
        Return the ASM/IR instruction
        
        :param adreg : Data register containing elements of the A component
        :type adreg : class:`asmgen.registers.data_reg`
        :param bdreg : Data register containing elements of the B component
        :type bdreg : class:`asmgen.registers.data_reg`
        :param cdreg : Data register containing elements of the C component
        :type cdreg : class:`asmgen.registers.data_reg`
        :param a_dt : Data type of the A component
        :type a_dt : class:`asmgen.registers.asm_data_type`
        :param b_dt : Data type of the B component
        :type b_dt : class:`asmgen.registers.asm_data_type`
        :param c_dt : Data type of the C component
        :type c_dt : class:`asmgen.registers.asm_data_type`
        :return : ASM/IR instruction corresponding to the operation
        :rtype : str
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    def check_triple(self, a_dt : adt, b_dt : adt, c_dt : adt):
        """
        Check if the operation supports the specified data type combination
        
        :param a_dt : Data type of the A component
        :type a_dt : class:`asmgen.registers.asm_data_type`
        :param b_dt : Data type of the B component
        :type b_dt : class:`asmgen.registers.asm_data_type`
        :param c_dt : Data type of the C component
        :type c_dt : class:`asmgen.registers.asm_data_type`
        :raises ValueError: if an unsupported datatype combination is passed
        """
        triple = adt_triple(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)
        if triple not in self.supported_triples():
            raise ValueError(f"Unsupported type combination a={a_dt},b={b_dt},c={c_dt}")

class dummy_opd3(opd3):
    """
    Dummy opd3 operation; ISAs assign this by default to operations they do not support
    """

    @property
    def widening_method(self) -> widening_method:
        raise NotImplementedError(self.NIE_MESSAGE)

    def supported_triples(self) -> list[adt_triple]:
        raise NotImplementedError(self.NIE_MESSAGE)

    def check_modifiers(self, modifiers : set[modifier]):
        raise NotImplementedError(self.NIE_MESSAGE)

    def __call__(self, *, adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                 a_dt : adt, b_dt : adt, c_dt : adt, modifiers : set[modifier], **kwargs) -> str:
        raise NotImplementedError(self.NIE_MESSAGE)
