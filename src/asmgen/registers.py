from abc import ABC,abstractmethod
from enum import Enum,auto

from typing import Self

#TODO: unify and deduplify handling different register types

class reg_tracker:
    def __init__(self, reg_type_init_list : list[tuple[str,int]] = []):
        # empty set
        # self.used_gregs : set[int] = set()
        # self.used_vregs : set[int] = set()
        # self.used_fregs : set[int] = set()
        # self.clobbered_gregs : set[int] = set()
        # self.clobbered_vregs : set[int] = set()
        # self.clobbered_fregs : set[int] = set()

        # self.max_gregs : int = max_greg
        # self.max_vregs : int = max_vreg
        # self.max_fregs : int = max_freg
        # self.gregs_available : list[int] = list(range(self.max_gregs))
        # self.vregs_available : list[int] = list(range(self.max_vregs))
        # self.fregs_available : list[int] = list(range(self.max_fregs))

        # self.aliased_gregs : dict[str,int] = {}
        # self.aliased_fregs : dict[str,int] = {}

        self.registered_types : set[str] = set()

        self.max_regs       : dict[str,int] = {}
        self.available_regs : dict[str,list[int]] = {}
        self.used_regs      : dict[str,set[int]] = {}
        self.clobbered_regs : dict[str,set[int]] = {}
        self.aliased_regs   : dict[str,dict[str,int]] = {}

        for tag,max_regs in reg_type_init_list:
            self.add_type(type_tag=tag, max_regs=max_regs)

    def add_type(self, type_tag : str, max_regs : int):
        if type_tag in self.registered_types:
            raise ValueError(f"Type already tracked: {type_tag}")
        self.registered_types.add(type_tag)
        self.max_regs[type_tag] = max_regs
        self.available_regs[type_tag] = list(range(max_regs))
        self.clobbered_regs[type_tag] = set()
        self.used_regs[type_tag] = set()
        self.aliased_regs[type_tag] = {}

    def reset(self):
        for tag in self.registered_types:
            self.clobbered_regs[tag] = set()
            self.used_regs[tag] = set()

    def reserve_any_reg(self, type_tag : str):
        if type_tag not in self.registered_types:
            raise ValueError(f"Type not tracked: {type_tag}")
        for idx in self.available_regs[type_tag]:
            if not idx in self.used_regs[type_tag]:
                self.used_regs[type_tag].add(idx)
                self.clobbered_regs[type_tag].add(idx)
                return idx
        raise IndexError(f"All {type_tag} registers in use!")

    def alias_reg(self, type_tag : str, name : str, idx : int):
        if type_tag not in self.registered_types:
            raise ValueError(f"Type not tracked: {type_tag}")
        if not idx in self.used_regs[type_tag]:
            raise IndexError(f"can't alias unused {type_tag} register nr. {idx}")
        # Throw error if an alias already exists.
        # Theoretically I don't see an issue using multiple aliases
        # for the same register, but I feel just allowing it will lead
        # to some nasty bugs
        if idx in self.aliased_regs[type_tag].values():
            alias_index = list(self.aliased_regs[type_tag].values()).index(idx)
            alias = list(self.aliased_regs[type_tag].keys())[alias_index]
            raise IndexError(f"can't alias {type_tag} register nr. {idx}, "
                             f"it already has the alias \"{alias}\"")
        self.aliased_regs[type_tag][name] = idx

    def reserve_specific_reg(self, type_tag : str, idx : int):
        if type_tag not in self.registered_types:
            raise ValueError(f"Type not tracked: {type_tag}")
        if idx in self.used_regs[type_tag]:
            raise IndexError(f"{type_tag} register nr. {idx} already in use")
        if idx >= self.max_regs[type_tag]:
            raise IndexError((f"{type_tag} register nr. {idx} not accessible"
                              f"(max_regs = {self.max_regs[type_tag]})"))

        self.used_regs[type_tag].add(idx)
        self.clobbered_regs[type_tag].add(idx)

    def unuse_reg(self, type_tag : str, idx : int):
        if type_tag not in self.registered_types:
            raise ValueError(f"Type not tracked: {type_tag}")
        # Check if an alias exists and remove it.
        # Purpose is to get an error if an alias is used
        # after the register was freed up
        if idx in self.aliased_regs[type_tag].values():
            # TODO: This is suspicious, I want a test
            alias_index = list(self.aliased_regs[type_tag].values()).index(idx)
            alias = list(self.aliased_regs[type_tag].keys())[alias_index]
            del self.aliased_regs[type_tag][alias]
        self.used_regs[type_tag].remove(idx)

    def available_reg_count(self, type_tag : str) -> int:
        return len(self.regs_available[type_tag])

    def used_reg_count(self, type_tag : str) -> int:
        return len(self.used_regs[type_tag])

    def get_used_regs(self, type_tag : str) -> set[int]:
        return self.used_regs[type_tag]

    def get_clobbered_regs(self, type_tag : str) -> set[int]:
        return self.clobbered_regs[type_tag]


class asm_data_type(Enum):
    FP8E4M3 = auto()
    FP8E5M2 = auto()
    UINT8   = auto()
    SINT8   = auto()
    BF16    = auto()
    HALF    = auto()
    FP16    = HALF
    UINT16  = auto()
    SINT16  = auto()
    TF32    = auto()
    XF32    = TF32
    SINGLE  = auto()
    FP32    = SINGLE
    UINT32  = auto()
    SINT32  = auto()
    DOUBLE  = auto()
    FP64    = DOUBLE
    UINT64  = auto()
    SINT64  = auto()

def adt_size(dt : asm_data_type) -> int:
    size_map = {
        asm_data_type.FP8E4M3 : 1,
        asm_data_type.FP8E5M2 : 1,
        asm_data_type.UINT8   : 1,
        asm_data_type.SINT8   : 1,
        asm_data_type.BF16    : 2,
        asm_data_type.HALF    : 2,
        asm_data_type.UINT16  : 2,
        asm_data_type.SINT16  : 2,
        asm_data_type.TF32    : 4,
        asm_data_type.SINGLE  : 4,
        asm_data_type.UINT32  : 4,
        asm_data_type.SINT32  : 4,
        asm_data_type.DOUBLE  : 8,
        asm_data_type.UINT64  : 8,
        asm_data_type.SINT64  : 8,
    }
    if dt not in size_map:
        raise RuntimeError(f"Invalid asm_data_type: {dt}")
    return size_map[dt]

def adt_is_float(dt : asm_data_type) -> bool:
    adt = asm_data_type
    fp_types = [adt.FP64, adt.FP32, adt.FP16, adt.BF16, adt.FP8E4M3, adt.FP8E5M2]
    return dt in fp_types

def adt_is_int(dt : asm_data_type) -> bool:
    adt = asm_data_type
    int_types = [adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8,
                 adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8]
    return dt in int_types

def adt_is_signed(dt : asm_data_type) -> bool:
    adt = asm_data_type
    int_types = [adt.SINT64, adt.SINT32, adt.SINT16, adt.SINT8]
    return dt in int_types

def adt_is_unsigned(dt : asm_data_type) -> bool:
    adt = asm_data_type
    int_types = [adt.UINT64, adt.UINT32, adt.UINT16, adt.UINT8]
    return dt in int_types

class adt_triple:
    def __init__(self,
                 a_dt : asm_data_type,
                 b_dt : asm_data_type,
                 c_dt : asm_data_type):
        self.a = a_dt
        self.b = b_dt
        self.c = c_dt
    def __eq__(self, other : Self) -> bool:
        return (self.a == other.a) and\
               (self.b == other.b) and\
               (self.c == other.c)
    def __repr__(self) -> str:
        return f"{str(self.a), str(self.b), str(self.c)}"

class asm_index_type(Enum):
    INT8    = 1
    INT16   = 2
    INT32   = 4
    INT64   = 8

class greg(ABC):
    @abstractmethod
    def __init__(self, reg_idx : int):
        raise NotImplementedError(NIE_MESSAGE)
    @abstractmethod
    def __str__(self):
        raise NotImplementedError(NIE_MESSAGE)

class data_reg(ABC):
    @abstractmethod
    def __init__(self, reg_idx : int):
        raise NotImplementedError(NIE_MESSAGE)
    @abstractmethod
    def __str__(self):
        raise NotImplementedError(NIE_MESSAGE)

class freg(data_reg):
    pass

class vreg(data_reg):
    pass

class treg(data_reg):
    pass
