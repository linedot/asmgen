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

class operand_restriction(Enum):
    IDXMIN        = auto() # There is a minimum allowed index
    IDXMAX        = auto() # There is a maximum allowed index
    IDXONEOF      = auto() # Index has to be element of a set
    IDXMULTIPLEOF = auto() # Index has to be a multiple of an integer (i.e 2: 2,4,6)
    IDXOTHERPLUSN = auto() # Index depends on index of another parameter and
                           # has to be other idx + n
    IDXOTHERPLUSNMOD = auto() # same as the one before but modulo a value (SVE)

class ArgumentDependencyError(Exception):
    def __init__(self, name: str, deps: list[str]):
        self.name = name
        self.deps = deps
        depstr = ", ".join(deps)
        self.msg = f"unset dependency(ies) for {name}: {depstr}"

        super().__init(self.msg)

class operand_constraint(ABC):

    type value_type = int|set[int]|tuple[str,int]|tuple[str,int,int]

    def modify_context(modifiers : set[Enum],
                       context : dict[str,value_type]) -> dict[str,value_type]:
        """
        modifies context based on specified modifiers and returns it,
        leaving original context unchanged. To be overriden by inheriting class.

        :param modifiers: modifiers to apply to the operation 
                          (like opd3_modifier.* or opdna1_modifier.*)
        :param context: dictionary of already assigned argument values
        """

        return context


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
        """

        ctx = context
        if modifiers:
            if type(modifiers[0]) == Enum:
                raise ValueError("Modfier type can't be a raw Enum")
            if any(type(m) != type(modifiers[0]) for m in modifiers):
                raise ValueError("All modifiers must be of the same type")
            ctx = self.modify_context(modifiers=modifiers, context=context)

        self.validate(name=name, val=val, context=ctx)

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
        """
        ctx = context
        if modifiers:
            if type(modifiers[0]) == Enum:
                raise ValueError("Modfier type can't be a raw Enum")
            if any(type(m) != type(modifiers[0]) for m in modifiers):
                raise ValueError("All modifiers must be of the same type")
            ctx = self.modify_context(modifiers=modifiers, context=context)

        for v in self.valid_values(name=name, context=ctx):
            yield v

    @abstractmethod
    def validate(self, name : str, val : value_type, context : dict[str,value_type]):
        """
        Checks if a value is valid for an argument and raises an Error if it is not.
        To be implemented by an inheriting class
        
        :param name: name of the argument (like 'adreg')
        :param context: dictionary of already assigned argument values

        :raises ArgumentDependencyError: If the operand depends on the value
                                         of another operand and it is unset
        :raises ValueError: if the value is invalid
        """

    @abstractmethod
    def valid_values(self, name : str, context : dict[str,value_type]):
        """
        Returns an iterable over valid values for a given operand. To
        be implemented by an inheriting class
        
        :param name: name of the argument (like 'adreg')
        :param context: dictionary of already assigned argument values

        :raises ArgumentDependencyError: If the operand depends on the value
                                         of another operand and it is unset
        """

def make_ord_prefix(i : int) -> 'str':
    if i > 25 or i < 0:
        raise ValueError("index outside of allowed range [0,25]")

    return chr(ord('a')+i)

class operation(ABC):
    """
    Abstraction over operations/instructions
    """

    NIE_MESSAGE="Inheriting class must implement this method"

    @abstractmethod
    def supported_dts(self) -> list[dict[str,adt]]:
        """
        Returns a list of valid datatypes for each argument
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    @abstractmethod
    def get_required_params(self, modifiers : set[Enum]) -> list[set[str]]:
        """
        Based on supplied modifiers, return a list of additional
        parameters
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    @abstractmethod
    def get_operand_restrictions(self, op : str) -> set[operand_restriction]:
        """
        For a specific operand get a set of required restrictions if any
        apply
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    @abstractmethod
    def get_operand_restriction_value(self, op : str,
                                      rstr : operand_restriction) \
      -> int|set[int]|tuple[str,int]|tuple[str,int,int]:
        """
        For a specific operand and restriction type, get the value for
        the restriction
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    @abstractmethod
    def check_modifiers(self, modifiers : set[Enum]):
        """
        Confirm validity of the supplied modifier combination
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    def check_dts(self, dts : dict[str,adt]):
        
        # <= on items is true if it's a subset
        if not any(dts.items() <= sup_dts.items() for sup_dts in self.supported_dts()):
            err_msg = "Invalid data type combination: "
            err_msg += ", ".join(f"{oprnd}:{dt.name}" for oprnd,dt in dts.items())
            raise ValueError(err_msg)

    def check_operand_restrictions(self,
                                   modifiers : set[Enum],
                                   kwargs : dict[str,int|greg_type|data_reg|adt]):

        for name, oprnd in kwargs.items():
            rstrs = self.get_operand_restrictions(name)
            if not rstrs:
                continue

            if operand_restriction.IDXMIN in rstrs:
                minval = self.get_operand_restriction_value(
                        op=name,
                        modifiers=modifiers,
                        rstr=operand_restriction.IDXMIN)
                if oprnd.idx < minval:
                    raise ValueError(f"{name} index must be >= {minval}")

            if operand_restriction.IDXMAX in rstrs:
                maxval = self.get_operand_restriction_value(
                        op=name,
                        modifiers=modifiers,
                        rstr=operand_restriction.IDXMAX)
                if oprnd.idx > maxval:
                    raise ValueError(f"{name} index must be <= {maxval}")

            if operand_restriction.IDXONEOF in rstrs:
                valset = self.get_operand_restriction_value(
                        op=name,
                        modifiers=modifiers,
                        rstr=operand_restriction.IDXONEOF)
                if oprnd.idx not in valset:
                    raise ValueError(f"{name} index must be one of {valset}")

            if operand_restriction.IDXMULTIPLEOF in rstrs:
                multiple = self.get_operand_restriction_value(
                        op=name,
                        modifiers=modifiers,
                        rstr=operand_restriction.IDXMULTIPLEOF)
                if 0 != (oprnd.idx % multiple):
                    raise ValueError(f"{name} index must be a multiple of {multiple}")

            if operand_restriction.IDXOTHERPLUSN in rstrs:
                other,offset = self.get_operand_restriction_value(
                        op=name,
                        modifiers=modifiers,
                        rstr=operand_restriction.IDXOTHERPLUSN)
                if oprnd.idx != kwargs[other].idx+offset:
                    raise ValueError(
                            f"{name} index must be index of {other} plus {offset}")

            if operand_restriction.IDXOTHERPLUSNMOD in rstrs:
                other,offset,modval = self.get_operand_restriction_value(
                        op=name,
                        modifiers=modifiers,
                        rstr=operand_restriction.IDXOTHERPLUSNMOD)
                if oprnd.idx != (kwargs[other].idx+offset) % modval:
                    raise ValueError(
                            (f"{name} index must be index of {other} "
                             f"plus {offset} modulo {modval}"))

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

        if len(dregs) < 1:
            raise ValueError("No dregs passed to opdna1 operation")

        self.check_modifiers(modifiers)
        self.check_dts(dts)

        extra_params = self.get_required_params(modifiers)
        for p in extra_params:
            params_specified = len(p.intersection(set(kwargs.keys())))
            if params_specified > 1:
                raise ValueError(f"{', '.join(sorted(p))} are mutually exclusive")
            if params_specified == 0:
                raise ValueError(f"Missing one of these parameters: {', '.join(sorted(p))}")

        # generate dreg args
        for i,reg in enumerate(dregs):
            pfx = make_ord_prefix(i)
            dreg_name = f"{pfx}dreg"
            dt_name = f"{pfx}_dt"
            kwargs[dreg_name] = reg
            kwargs[dt_name] = dts[dreg_name]

        for i,reg in enumerate(gregs):
            pfx = make_ord_prefix(i)
            greg_name = f"{pfx}greg"
            kwargs[greg_name] = reg

        
        self.check_operand_restrictions(modifiers=modifiers, kwargs=kwargs)

        kwargs['modifiers'] = modifiers
        
        # opdna1 has a different interface
        kwargs['dregs'] = dregs

        return self.implementation(**kwargs)


    @abstractmethod
    def implementation(self, **kwargs) -> str:
        """
        Actual implementation of the operation to be implemented
        by the inheriting class
        """
        raise NotImplementedError(self.NIE_MESSAGE)

class widening_method(Enum):
    """
    Possible methods ISAs can have for widening instructions
    """
    NONE = auto()
    VEC_GROUP = auto()
    VEC_MULTI = auto()
    DOT_NEIGHBOURS = auto()
    SPLIT_INSTRUCTIONS = auto()

class opd3_modifier(Enum):
    """
    Possible modifiers for an opd3 instruction/operation
    """
    NP = auto()
    IDX = auto()
    REGIDX = auto()
    PART = auto()
    VF = auto()
    MASK = auto()


class opd3(operation):
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
    def check_modifiers(self, modifiers : set[opd3_modifier]):
        """
        Checks whether the operations supports the specified modifiers


        :param modifiers: set containing the modifiers to check
        :type modifiers: set[class:`asmgen.asmblocks.operations.opd3_modifier`]
        :raises ValueError: If an unsupported modifier is in the specified set
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    def __call__(self, *,
                 adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                 a_dt : adt, b_dt : adt, c_dt : adt,
                 modifiers : set[opd3_modifier] = set(),
                 **kwargs) -> str:
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

        return self.execute(
            dregs=[adreg,bdreg,cdreg],
            gregs=[],
            dts={'adreg':a_dt,'bdreg':b_dt,'cdreg':c_dt},
            modifiers=modifiers,
            **kwargs
        )


class dummy_opd3(opd3):
    """
    Dummy opd3 operation; ISAs assign this by default to operations they do not support
    """

    @property
    def widening_method(self) -> widening_method:
        raise NotImplementedError(self.NIE_MESSAGE)

    def supported_dts(self) -> list[dict[str,adt]]:
        raise NotImplementedError(self.NIE_MESSAGE)

    def check_modifiers(self, modifiers : set[opd3_modifier]):
        raise NotImplementedError(self.NIE_MESSAGE)

    def get_required_params(self, modifiers : set[opd3_modifier]) -> list[set[str]]:
        raise NotImplementedError(self.NIE_MESSAGE)

    def get_operand_restrictions(self, op : str) -> set[operand_restriction]:
        raise NotImplementedError(self.NIE_MESSAGE)

    def get_operand_restriction_value(self, op : str,
                                      rstr : operand_restriction) -> int:
        raise NotImplementedError(self.NIE_MESSAGE)

    def implementation(self, *,
                       adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                       a_dt : adt, b_dt : adt, c_dt : adt,
                       modifiers : set[opd3_modifier], **kwargs) -> str:
        raise NotImplementedError(self.NIE_MESSAGE)

class opdna1_modifier(Enum):
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

    @abstractmethod
    def supported_dts(self) -> list[dict[str,adt]]:
        """
        Return the list of supported data types
        
        :return : list of supported data types
        :rtype : class:`asmgen.registers.asm_data_type`
        """
        raise NotImplementedError(self.NIE_MESSAGE)

    @abstractmethod
    def check_modifiers(self, modifiers : set[opdna1_modifier]):
        """
        Checks whether the operations supports the specified modifiers


        :param modifiers: set containing the modifiers to check
        :type modifiers: set[class:`asmgen.asmblocks.operations.opd1a1_modifier`]
        :raises ValueError: If an unsupported modifier is in the specified set
        """
        raise NotImplementedError(self.NIE_MESSAGE)

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

    def supported_dts(self) -> list[adt_triple]:
        raise NotImplementedError(self.NIE_MESSAGE)

    def check_modifiers(self, modifiers : set[opdna1_modifier]):
        raise NotImplementedError(self.NIE_MESSAGE)

    def get_required_params(self, modifiers : set[opdna1_modifier]) -> list[str]:
        raise NotImplementedError(self.NIE_MESSAGE)

    def get_operand_restrictions(self, op : str) -> set[operand_restriction]:
        raise NotImplementedError(self.NIE_MESSAGE)

    def get_operand_restriction_value(self, op : str,
                                      rstr : operand_restriction) -> int:
        raise NotImplementedError(self.NIE_MESSAGE)

    def implementation(self, *,
                       dregs : list[data_reg], agreg : greg_type,
                       a_dt : adt,
                       modifiers : set[opdna1_modifier], **kwargs) -> str:
        raise NotImplementedError(self.NIE_MESSAGE)
