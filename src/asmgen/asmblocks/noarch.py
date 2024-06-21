from abc import ABC, abstractmethod
from enum import Enum, unique

import sys
if not sys.version_info >= (3, 10):
    from typing_extensions import TypeAlias
else:
    from typing import TypeAlias



#TODO: unify and deduplify handling different register types

class reg_tracker:
    def __init__(self, max_greg : int, max_vreg : int, max_freg : int):
        # empty set
        self.used_gregs : set[int] = set()
        self.used_vregs : set[int] = set()
        self.used_fregs : set[int] = set()
        self.clobbered_gregs : set[int] = set()
        self.clobbered_vregs : set[int] = set()
        self.clobbered_fregs : set[int] = set()

        self.max_gregs : int = max_greg
        self.max_vregs : int = max_vreg
        self.max_fregs : int = max_freg
        self.gregs_available : list[int] = [i for i in range(self.max_gregs)]
        self.vregs_available : list[int] = [i for i in range(self.max_vregs)]
        self.fregs_available : list[int] = [i for i in range(self.max_fregs)]

        self.aliased_gregs : dict[str,int] = {}
        self.aliased_fregs : dict[str,int] = {}

    def reset(self):
        self.used_gregs = set()
        self.used_vregs = set()
        self.used_fregs = set()
        self.clobbered_gregs = set()
        self.clobbered_vregs = set()
        self.clobbered_fregs = set()

    def reserve_any_greg(self):
        for i in self.gregs_available:
            if not i in self.used_gregs:
                self.used_gregs.add(i)
                self.clobbered_gregs.add(i)
                return i
        raise IndexError(f"All gp registers in use!")

    def reserve_any_freg(self):
        for i in self.fregs_available:
            if not i in self.used_fregs:
                self.used_fregs.add(i)
                self.clobbered_fregs.add(i)
                return i
        raise IndexError(f"All fp registers in use!")

    def alias_greg(self, name : str, i : int):
        if not i in self.used_gregs:
            raise IndexError(f"can't alias unused gp register nr. {i}")
        # Throw error if an alias already exists.
        # Theoretically I don't see an issue using multiple aliases
        # for the same register, but I feel just allowing it will lead
        # to some nasty bugs
        if i in self.aliased_gregs.values():
            alias_index = list(self.aliased_gregs.values()).index(i)
            alias = list(self.aliased_gregs.keys())[alias_index]
            raise IndexError(f"can't alias gp register nr. {i}, it already has the alias \"{alias}\"")
        self.aliased_gregs[name] = i

    def alias_freg(self, name : str, i : int):
        if not i in self.used_fregs:
            raise IndexError(f"can't alias unused fp register nr. {i}")
        # Throw error if an alias already exists.
        # Theoretically I don't see an issue using multiple aliases
        # for the same register, but I feel just allowing it will lead
        # to some nasty bugs
        if i in self.aliased_fregs.values():
            alias_index = list(self.aliased_fregs.values()).index(i)
            alias = list(self.aliased_fregs.keys())[alias_index]
            raise IndexError(f"can't alias fp register nr. {i}, it already has the alias \"{alias}\"")
        self.aliased_fregs[name] = i

    def reserve_specific_greg(self, i : int):
        if i in self.used_gregs:
            raise IndexError(f"fp register nr. {i} already in use")

        if i >= self.max_gregs:
            raise IndexError(f"fp register nr. {i} not accessible (max_gregs = {self.max_gregs})")

        self.used_gregs.add(i)
        self.clobbered_gregs.add(i)

    def reserve_specific_freg(self, i : int):
        if i in self.used_fregs:
            raise IndexError(f"gp register nr. {i} already in use")

        if i >= self.max_fregs:
            raise IndexError(f"gp register nr. {i} not accessible (max_fregs = {self.max_fregs})")

        self.used_fregs.add(i)
        self.clobbered_fregs.add(i)

    def reserve_any_vreg(self):
        for i in self.vregs_available:
            if not i in self.used_vregs:
                self.used_vregs.add(i)
                self.clobbered_vregs.add(i)
                return i
        raise IndexError(f"All vec registers in use!")

    def reserve_specific_vreg(self, i : int):
        if i in self.used_vregs:
            raise IndexError(f"vec register nr. {i} already in use")

        if i >= self.max_vregs:
            raise IndexError(f"vec register nr. {i} not accessible (max_vregs = {self.max_vregs})")

        self.used_vregs.add(i)
        self.clobbered_vregs.add(i)

    def unuse_greg(self, i : int):
        # Check if an alias exists and remove it.
        # Purpose is to get an error if an alias is used
        # after the register was freed up
        if i in self.aliased_gregs.values():
            # TODO: This is suspicious, I want a test
            alias_index = list(self.aliased_gregs.values()).index(i)
            alias = list(self.aliased_gregs.keys())[alias_index]
            del self.aliased_gregs[alias]
        self.used_gregs.remove(i)

    def unuse_freg(self, i : int):
        # Check if an alias exists and remove it.
        # Purpose is to get an error if an alias is used
        # after the register was freed up
        if i in self.aliased_fregs.values():
            # TODO: This is suspicious, I want a test
            alias_index = list(self.aliased_fregs.values()).index(i)
            alias = list(self.aliased_fregs.keys())[alias_index]
            del self.aliased_fregs[alias]
        self.used_fregs.remove(i)

    def unuse_vreg(self, i : int):
        self.used_vregs.remove(i)

    def gregs_available_count(self) -> int:
        return len(self.gregs_available)

    def vregs_available_count(self) -> int:
        return len(self.vregs_available)

    def fregs_available_count(self) -> int:
        return len(self.fregs_available)

    def gregs_used_count(self) -> int:
        return len(self.used_gregs)

    def vregs_used_count(self) -> int:
        return len(self.used_vregs)

    def fregs_used_count(self) -> int:
        return len(self.used_fregs)

    def get_used_gregs(self) -> set[int]:
        return self.used_gregs

    def get_used_vregs(self) -> set[int]:
        return self.used_vregs

    def get_used_fregs(self) -> set[int]:
        return self.used_fregs

    def get_clobbered_gregs(self) -> set[int]:
        return self.clobbered_gregs

    def get_clobbered_vregs(self) -> set[int]:
        return self.clobbered_vregs

    def get_clobbered_fregs(self) -> set[int]:
        return self.clobbered_fregs

        

class asm_data_type(Enum):
    HALF   = 2
    FP16   = 2
    SINGLE = 4
    FP32   = 4
    DOUBLE = 8
    FP64   = 8

class asm_index_type(Enum):
    INT8    = 1
    INT16   = 2
    INT32   = 4
    INT64   = 8

class greg(ABC):
    @abstractmethod
    def __init__(self, reg_idx : int):
        raise NotImplementedError("Method to be implemented by derived class")
    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Method to be implemented by derived class")

class freg(ABC):
    @abstractmethod
    def __init__(self, reg_idx : int):
        raise NotImplementedError("Method to be implemented by derived class")
    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Method to be implemented by derived class")

class vreg(ABC):
    @abstractmethod
    def __init__(self, reg_idx : int):
        raise NotImplementedError("Method to be implemented by derived class")
    @abstractmethod
    def __str__(self):
        raise NotImplementedError("Method to be implemented by derived class")

class asmgen(ABC):

    greg_type : TypeAlias = greg
    freg_type : TypeAlias = freg
    vreg_type : TypeAlias = vreg

    def __init__(self):
        self.output_inline:bool = True

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
    def data_size(data_type : asm_data_type):
        return data_type.value

    @abstractmethod
    def supportedby_cpuinfo(self, cpuinfo : str) -> bool:
        raise NotImplementedError("Method to be implemented by derived class")

    def supported_on_host(self) -> bool:
        # TODO: maybe something more cross-platform?
        f = open("/proc/cpuinfo","r")
        cpuinfo = f.read()
        f.close()
        return self.supportedby_cpuinfo(cpuinfo)

    @abstractmethod
    def greg(self, reg_idx : int) -> greg:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def freg(self, reg_idx : int) -> freg:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def vreg(self, reg_idx : int) -> vreg:
        raise NotImplementedError("Method to be implemented by derived class")

    @property
    @abstractmethod
    def simd_size(self) -> int:
        raise NotImplementedError("Method to be implemented by derived class")

    @property
    @abstractmethod
    def c_simd_size_function(self) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @property
    @abstractmethod
    def is_vla(self) -> bool:
        raise NotImplementedError("Method to be implemented by derived class")

    @property
    @abstractmethod
    def are_fregs_in_vregs(self) -> bool:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def indexable_elements(self, datatype : asm_data_type) -> int:
        raise NotImplementedError("Method to be implemented by derived class")

    @property
    @abstractmethod
    def max_vregs(self) -> int:
        raise NotImplementedError("Method to be implemented by derived class")

    @property
    @abstractmethod
    def max_fregs(self) -> int:
        raise NotImplementedError("Method to be implemented by derived class")

    @property
    @abstractmethod
    def max_gregs(self) -> int:
        raise NotImplementedError("Method to be implemented by derived class")

    @property
    @abstractmethod
    def min_prefetch_offset(self) -> int:
        raise NotImplementedError("Method to be implemented by derived class")

    @property
    @abstractmethod
    def max_prefetch_offset(self) -> int:
        raise NotImplementedError("Method to be implemented by derived class")

    @property
    @abstractmethod
    def min_load_voff(self) -> int:
        raise NotImplementedError("Method to be implemented by derived class")

    @property
    @abstractmethod
    def max_load_voff(self) -> int:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def min_load_immoff(self, datatype : asm_data_type) -> int:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def max_load_immoff(self, datatype : asm_data_type) -> int:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def max_fload_immoff(self, datatype : asm_data_type) -> int:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def label(self, label : str) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def jump(self, label : str) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def jzero(self, reg : greg_type, label : str) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def jfzero(self, freg1 : freg_type, freg2 : freg_type, 
               greg : greg_type, label : str, 
               datatype : asm_data_type) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def jvzero(self, vreg1 : vreg_type, freg : freg_type,
               vreg2 : vreg_type, greg : greg_type, label : str,
               datatype : asm_data_type) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def loopbegin(self, reg : greg_type, label : str) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def loopbegin_nz(self, reg : greg_type, label : str, labelskip : str) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def loopend(self, reg : greg_type, label : str) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod 
    def fma(self, avreg : vreg_type, bvreg : vreg_type, cvreg : vreg_type,
            datatype : asm_data_type) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod 
    def fmul(self, avreg : vreg_type, bvreg : vreg_type, cvreg : vreg_type,
             datatype : asm_data_type) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def fma_idx(self, avreg : vreg_type, bvreg : vreg_type, cvreg : vreg_type,
                idx : int, datatype : asm_data_type) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def fma_vf(self, avreg : vreg_type, bfreg : freg_type, cvreg : vreg_type,
               datatype : asm_data_type) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod 
    def fmul_vf(self, avreg : vreg_type, bfreg : freg_type, cvreg : vreg_type,
                datatype : asm_data_type) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def zero_greg(self, greg : greg_type) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def zero_freg(self, freg : freg_type) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def zero_vreg(self, vreg : vreg_type, datatype : asm_data_type) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def mov_greg(self, src : greg_type, dst : greg_type) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def mov_freg(self, src : freg_type, dst : freg_type, datatype : asm_data_type) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def mov_greg_to_param(self, reg : greg_type, param : str) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def mov_param_to_greg(self, param : str, dst : greg_type) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def mov_param_to_greg_shift(self, param : str, dst : greg_type,
                                offset : int) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def mov_greg_imm(self, reg : greg_type, imm : int) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def add_greg_imm(self, reg : greg_type, offset : int) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @property
    @abstractmethod
    def has_add_greg_voff(self) -> bool:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def add_greg_voff(self, reg : greg_type, offset : int,
                      datatype : asm_data_type) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def add_greg_greg(self, dst : greg_type, reg1 : greg_type, reg2 : greg_type) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def shift_greg_left(self, reg : greg_type, offset : int) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def shift_greg_right(self, reg : greg_type, offset : int) -> str:
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def prefetch_l1_boff(self, areg : greg_type, offset : int):
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def load_pointer(self, areg : greg_type, name : str):
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def load_scalar_immoff(self, areg : greg_type, offset : int,
                           freg : freg_type, datatype : asm_data_type):
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def load_vector_voff(self, areg : greg_type, voffset : int, 
                         vreg : vreg_type, datatype : asm_data_type):
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def load_vector(self, areg : greg_type, ignored_offset : int,
                    vreg : vreg_type, datatype : asm_data_type):
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def load_vector_immstride(self, areg : greg_type, byte_stride : int,
                    vreg : vreg_type, datatype : asm_data_type):
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def load_vector_gregstride(self, areg : greg_type, sreg : greg_type,
                    vreg : vreg_type, datatype : asm_data_type):
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def load_vector_gather(self, areg : greg_type, offvreg : vreg_type,
                           vreg : vreg_type, datatype : asm_data_type,
                           indextype : asm_index_type):
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def load_vector_dist1(self, areg : greg_type, ignored_offset : int,
                          vreg : vreg_type, datatype : asm_data_type):
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def load_vector_dist1_boff(self, areg : greg_type, offset : int, 
                               vreg : vreg_type, datatype : asm_data_type):
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def load_vector_dist1_inc(self, areg : greg_type, ignored_offset : int,
                              vreg : vreg_type, datatype : asm_data_type):
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def store_vector(self, areg : greg_type, ignored_offset : int,
                     vreg : vreg_type, datatype : asm_data_type):
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def store_vector_voff(self, areg : greg_type, voffset : int, 
                          vreg : vreg_type, datatype : asm_data_type):
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def store_vector_immstride(self, areg : greg_type, byte_stride : int,
                    vreg : vreg_type, datatype : asm_data_type):
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def store_vector_gregstride(self, areg : greg_type, sreg : greg_type,
                    vreg : vreg_type, datatype : asm_data_type):
        raise NotImplementedError("Method to be implemented by derived class")

    @abstractmethod
    def store_vector_scatter(self, areg : greg_type, offvreg : vreg_type,
                             vreg : vreg_type, datatype : asm_data_type,
                             indextype : asm_index_type):
        raise NotImplementedError("Method to be implemented by derived class")
