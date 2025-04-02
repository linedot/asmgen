from abc import ABC, abstractmethod
from enum import Enum,auto


from typing import TypeAlias

from .operations import opd3,dummy_opd3
from ..registers import (
    reg_tracker,
    greg,freg,vreg,treg,
    asm_data_type,
    adt_size,
    asm_index_type
)

class asmgen(ABC):
    """
    Abstract interface for asm code generator
    """
    greg_type : TypeAlias = greg
    freg_type : TypeAlias = freg
    vreg_type : TypeAlias = vreg
    treg_type : TypeAlias = treg

    def __init__(self):
        self.output_inline:bool = True
        self.fopa = dummy_opd3()
        self.fma = dummy_opd3()

    def set_output_inline(self, yesno : bool):
        self.output_inline = yesno

    def asmwrap(self, code : str) -> str:
        if self.output_inline:
            return f"\"{code}\\n\\t\"\n"
        else:
            return f"{code}\n"

    @staticmethod
    def operands(inputs : list[tuple[str,str,str]],
                 outputs : list[tuple[str,str,str]],
                 clobber : list) -> str:
        opblock  = ": "
        opblock += ",".join([f"[{n}] \"{t}\" ({init})" for n,t,init in outputs])
        opblock += "\n"
        opblock += ": "
        opblock += ",".join([f"[{n}] \"{t}\" ({init})" for n,t,init in inputs])
        opblock += "\n"
        opblock += ": "
        opblock += ",".join([f"\"{reg}\"" for reg in clobber])
        return opblock

    @staticmethod
    def data_size(dt : asm_data_type):
        return adt_size(dt)

    @abstractmethod
    def isaquirks(self, rt : reg_tracker, dt : asm_data_type):
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def supportedby_cpuinfo(self, cpuinfo : str) -> bool:
        raise NotImplementedError(NIE_MESSAGE)

    def supported_on_host(self) -> bool:
        # TODO: maybe something more cross-platform?
        f = open("/proc/cpuinfo","r")
        cpuinfo = f.read()
        f.close()
        return self.supportedby_cpuinfo(cpuinfo)

    @abstractmethod
    def greg(self, reg_idx : int) -> greg:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def freg(self, reg_idx : int) -> freg:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def vreg(self, reg_idx : int) -> vreg:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def treg(self, reg_idx : int) -> treg:
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def simd_size(self) -> int:
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def c_simd_size_function(self) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def is_vla(self) -> bool:
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def are_fregs_in_vregs(self) -> bool:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def indexable_elements(self, dt : asm_data_type) -> int:
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def max_tregs(self, dt : asm_data_type) -> int:
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def max_vregs(self) -> int:
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def max_fregs(self) -> int:
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def max_gregs(self) -> int:
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def min_prefetch_offset(self) -> int:
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def max_prefetch_offset(self) -> int:
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def min_load_voff(self) -> int:
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def max_load_voff(self) -> int:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def min_load_immoff(self, dt : asm_data_type) -> int:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def max_load_immoff(self, dt : asm_data_type) -> int:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def max_fload_immoff(self, dt : asm_data_type) -> int:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def label(self, label : str) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def jump(self, label : str) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def jzero(self, reg : greg_type, label : str) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def jfzero(self, freg1 : freg_type, freg2 : freg_type,
               greg : greg_type, label : str,
               dt : asm_data_type) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def jvzero(self, vreg1 : vreg_type, freg : freg_type,
               vreg2 : vreg_type, greg : greg_type, label : str,
               dt : asm_data_type) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def loopbegin(self, reg : greg_type, label : str) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def loopbegin_nz(self, reg : greg_type, label : str, labelskip : str) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def loopend(self, reg : greg_type, label : str) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def zero_greg(self, greg : greg_type) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def zero_freg(self, freg : freg_type) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def zero_vreg(self, vreg : vreg_type, dt : asm_data_type) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def zero_treg(self, treg : treg_type, dt : asm_data_type) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def mov_greg(self, src : greg_type, dst : greg_type) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def mov_freg(self, src : freg_type, dst : freg_type, dt : asm_data_type) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def mov_greg_to_param(self, reg : greg_type, param : str) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def mov_param_to_greg(self, param : str, dst : greg_type) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def mov_param_to_greg_shift(self, param : str, dst : greg_type,
                                offset : int) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def mov_greg_imm(self, reg : greg_type, imm : int) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def mul_greg_imm(self, src : greg_type, dst : greg_type, offset : int) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def add_greg_imm(self, reg : greg_type, offset : int) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @property
    @abstractmethod
    def has_add_greg_voff(self) -> bool:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def add_greg_voff(self, reg : greg_type, offset : int,
                      dt : asm_data_type) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def add_greg_greg(self, dst : greg_type, reg1 : greg_type, reg2 : greg_type) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def shift_greg_left(self, reg : greg_type, offset : int) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def shift_greg_right(self, reg : greg_type, offset : int) -> str:
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def prefetch_l1_boff(self, areg : greg_type, offset : int):
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_pointer(self, areg : greg_type, name : str):
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_scalar_immoff(self, areg : greg_type, offset : int,
                           freg : freg_type, dt : asm_data_type):
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_vector_voff(self, areg : greg_type, voffset : int,
                         vreg : vreg_type, dt : asm_data_type):
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_vector(self, areg : greg_type, ignored_offset : int,
                    vreg : vreg_type, dt : asm_data_type):
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_vector_immstride(self, areg : greg_type, byte_stride : int,
                    vreg : vreg_type, dt : asm_data_type):
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_vector_gregstride(self, areg : greg_type, sreg : greg_type,
                    vreg : vreg_type, dt : asm_data_type):
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_vector_gather(self, areg : greg_type, offvreg : vreg_type,
                           vreg : vreg_type, dt : asm_data_type,
                           indextype : asm_index_type):
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_vector_dist1(self, areg : greg_type, ignored_offset : int,
                          vreg : vreg_type, dt : asm_data_type):
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_vector_dist1_boff(self, areg : greg_type, offset : int,
                               vreg : vreg_type, dt : asm_data_type):
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def load_vector_dist1_inc(self, areg : greg_type, ignored_offset : int,
                              vreg : vreg_type, dt : asm_data_type):
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def store_vector(self, areg : greg_type, ignored_offset : int,
                     vreg : vreg_type, dt : asm_data_type):
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def store_vector_voff(self, areg : greg_type, voffset : int,
                          vreg : vreg_type, dt : asm_data_type):
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def store_vector_immstride(self, areg : greg_type, byte_stride : int,
                    vreg : vreg_type, dt : asm_data_type):
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def store_vector_gregstride(self, areg : greg_type, sreg : greg_type,
                    vreg : vreg_type, dt : asm_data_type):
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def store_vector_scatter(self, areg : greg_type, offvreg : vreg_type,
                             vreg : vreg_type, dt : asm_data_type,
                             indextype : asm_index_type):
        raise NotImplementedError(NIE_MESSAGE)

    @abstractmethod
    def store_tile(self, areg : greg_type, ignored_offset : int,
                   treg : treg_type, dt : asm_data_type):
        raise NotImplementedError(NIE_MESSAGE)
