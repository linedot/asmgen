from abc import ABC,abstractmethod
from enum import Enum,auto

from typing import Self

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
            raise IndexError(f"can't alias gp register nr. {i}, "
                             f"it already has the alias \"{alias}\"")
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
            raise IndexError(f"can't alias fp register nr. {i}, "
                             f"it already has the alias \"{alias}\"")
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
