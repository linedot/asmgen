from abc import ABC,abstractmethod
from enum import Enum,auto

from ..registers import (
    data_reg,
    greg,freg,vreg,treg,
    asm_data_type as adt,
    adt_triple,
)

class widening_method(Enum):
    none = auto()
    vec_group = auto()
    vec_multi = auto()
    dot_neighbours = auto()
    split_instructions = auto()

#TODO: masked modifier
class modifier(Enum):
    np = auto()
    idx = auto()
    regidx = auto()
    part = auto()
    vf = auto()

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
        raise NotImplementedError(self.NIE_MESSAGE)

    @abstractmethod
    def supported_triples(self) -> list[adt_triple]:
        raise NotImplementedError(self.NIE_MESSAGE)

    @abstractmethod
    def __call__(self, adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                 a_dt : adt, b_dt : adt, c_dt : adt, 
                 modifiers : set[modifier], **kwargs) -> str:
        raise NotImplementedError(self.NIE_MESSAGE)

    def check_triple(self, a_dt : adt, b_dt : adt, c_dt : adt):
        triple = adt_triple(a_dt=a_dt, b_dt=b_dt, c_dt=c_dt)
        if triple not in self.supported_triples():
            raise ValueError("Unsupported type combination")

class dummy_opd3(opd3):

    @property
    def widening_method(self) -> widening_method:
        raise NotImplementedError(self.NIE_MESSAGE)

    def supported_triples(self) -> list[adt_triple]:
        raise NotImplementedError(self.NIE_MESSAGE)

    def __call__(self, adreg : data_reg, bdreg : data_reg, cdreg : data_reg,
                 a_dt : adt, b_dt : adt, c_dt : adt) -> str:
        raise NotImplementedError(self.NIE_MESSAGE)
