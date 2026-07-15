# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------

from enum import Enum,auto

from abc import abstractmethod

from .operation import operation
from .modifier import operation_modifier
from .misc import make_ord_prefix

from ...registers import asm_data_type as adt, data_reg

class opdna1_modifier(operation_modifier):
    """
    Possible modifiers for an instruction/operation
    """
    ILANE = auto()   # select lane with an immediate
    GLANE = auto()   # select lane with a greg
    TOFFSET = auto() # 2D offset in number of tiles
    VOFFSET = auto() # 1D offset in number of vectors
    IOFFSET = auto() # 1D offset in number of elements
    GOFFSET = auto() # 1D offset given by greg
    TINDEX = auto()  # 2D tile contains per-element indices/offsets
    VINDEX = auto()  # 1D vector contains per-element indices/offsets
    GSTRIDE = auto() # stride between elements given by greg
    ISTRIDE = auto() # stride between elements given by immediate
    POSTINC = auto() # increment address greg after operation
    STRUCT = auto()  # load a structure with multiple components 
                     # (i.e [Re,Im], [x,y,z] or [r,g,b,a])
    BCAST = auto()   # Broadcast one value into all lanes
    MASK  = auto()   # Masked operation
    ROW = auto()     # Row of a treg
    COL = auto()     # Column of a treg
    NT  = auto()     # Non-temporal ld/st

class opdna1_action(Enum):
    """
    Possible opdna1 actions
    """
    LOAD = auto()
    STORE = auto()

class opdna1(operation):
    """
    Assembly/IR instruction with n data operand and 1 address operand

    Absraction for loads/stores (maybe also prefetches)
    """
    NIE_MESSAGE="Method not implemented"

    def __call__(self, *, dregs : list[data_reg],
                 areg : greg_type, dt : adt,
                 modifiers : set[opdna1_modifier], **kwargs) -> str:
        """
        Return the ASM/IR instruction
        
        :param dregs : Data registers
        :type dregs : list[class:`asmgen.registers.data_reg`]
        :param areg : Address register
        :type areg : class:`asmgen.registers.greg_type`
        :param dt : Data type
        :type dt : class:`asmgen.registers.asm_data_type`
        :return : ASM/IR instruction corresponding to the operation
        :rtype : str
        """

        return self.execute(
            dregs=dregs,
            gregs=[areg],
            dts={
                make_ord_prefix(i)+'dreg' : dt for i in range(len(dregs))
            },
            modifiers=modifiers,
            **kwargs
        )

    @abstractmethod
    def implementation(self, *, dregs : list[data_reg],
                       agreg : greg_type, a_dt : adt,
                       modifiers : set[opdna1_modifier], **kwargs) -> str:
        raise NotImplementedError(self.NIE_MESSAGE)


class dummy_opdna1(opdna1):
    """
    Dummy opd1a1 operation; ISAs assign this by default to operations they do not support
    """

    def implementation(self, *,
                       dregs : list[data_reg], agreg : greg_type,
                       a_dt : adt,
                       modifiers : set[opdna1_modifier], **kwargs) -> str:
        raise NotImplementedError(self.NIE_MESSAGE)
