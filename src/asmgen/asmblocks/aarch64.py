# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
ARM64/AArch64 asm generator and related types
"""

from copy import deepcopy
from typing import Union

from .noarch import asmgen,comparison
from ..registers import (
    asm_data_type as adt,
    greg_base, freg_base
)

from .types.aarch64_types import aarch64_greg,aarch64_freg
from ..callconv.callconv import callconv


class aarch64(asmgen):
    """
    ARM64/AArch64 asmgen implementation
    """

    dt_greg_pfx = {
            adt.DOUBLE : "x",
            adt.SINGLE : "w",
            }

    cb_insts = {
            'nz' : 'cbnz',
            'ez' : 'cbz',
            'ne' : 'b.ne',
            'eq' : 'b.eq',
            'le' : 'b.le',
            'ge' : 'b.ge',
            'lt' : 'b.lt',
            'gt' : 'b.gt',
            }

    def __init__(self):
        super().__init__()
        self.callconvs = {
            "aapcs64" : callconv(
                param_regs={ 'greg' : list(range(0, 8)),
                             'freg' : list(range(0, 8))},
                caller_save_lists={ 'greg' : list(range(0, 8)) + # func args/return vals
                                             [8] + # XR, indirect result reg
                                             list(range(9,16)) + # locals
                                             [18] # platform reg
                                        ,
                                    'freg' : list(range(0, 8)) +
                                             list(range(16,32))
                                   },
                callee_save_lists={ 'greg' : list(range(19,29)) + # Callee-saved
                                             [29, # Frame Pointer
                                              30  # Link Register
                                              ] +
                                             [31], # SP
                                    'freg' : list(range(8,16))},
                spreg=31)
            }
        self.default_callconv = "aapcs64"

    def create_callconv(self, name : str = "default"):

        if "default" == name:
            name = self.default_callconv

        return deepcopy(self.callconvs[name])

    def get_parameters(self) -> list[str]:
        return []

    def set_parameter(self, name : str, value : Union[str,int]):
        raise ValueError(f"Invalid name {name} or value {value}")

    @property
    def are_fregs_in_vregs(self) -> bool:
        return True

    def label(self, *, label : str) -> str:
        return self.asmwrap(f"{self.labelstr(label)}:")

    def jump(self, *, label : str) -> str:
        return self.asmwrap(f"b {self.labelstr(label)}")

    def loopbegin(self, *, reg : greg_base,
                  label : str) -> str:
        asmblock  = self.asmwrap(f"{self.labelstr(label)}:")
        asmblock += self.asmwrap(f"sub {reg},{reg},1")
        return asmblock

    def loopbegin_nz(self, *, reg : greg_base,
                     label : str,
                     labelskip : str) -> str:
        asmblock  = self.asmwrap(f"cmp {reg},0")
        asmblock += self.asmwrap(f"b.eq {self.labelstr(labelskip)}")
        asmblock += self.asmwrap(f"{self.labelstr(label)}:")
        asmblock += self.asmwrap(f"sub {reg},{reg},1")
        return asmblock

    def loopend(self, *, reg : greg_base,
                label : str) -> str:
        asmblock  = self.asmwrap(f"cmp {reg},0")
        asmblock += self.asmwrap(f"b.ne {self.labelstr(label)}")
        return asmblock

    def cb(self, *, reg1: greg_base, reg2: greg_base,
           cmp: comparison, label: str) -> str:
        inst = self.cb_insts[cmp.name]
        if reg2 is None:
            return self.asmwrap(f"{inst} {reg1},{self.labelstr(label)}")
        else:
            return self.asmwrap(f"cmp {reg2},{reg1}")+\
                   self.asmwrap(f"{inst} {self.labelstr(label)}")

    def jzero(self, *, reg : greg_base,
              label : str) -> str:
        asmblock  = self.asmwrap(f"cmp {reg},0")
        asmblock += self.asmwrap(f"b.eq {self.labelstr(label)}")
        return asmblock

    def jfzero(self, *, freg1 : freg_base, freg2 : freg_base,
               greg : greg_base, label : str,
               dt : adt):
        asmblock  = self.asmwrap(f"fcmp {freg1},#0.0")
        asmblock += self.asmwrap(f"b.eq {self.labelstr(label)}")
        return asmblock

    @property
    def max_gregs(self):
        return 32

    @property
    def max_fregs(self):
        return 32

    def mov_greg(self, *, src : greg_base, dst : greg_base) -> str:
        return self.asmwrap(f"mov {dst},{src}")

    def mov_freg(self, *, src : freg_base, dst : freg_base,
                 dt : adt) -> str:
        return self.asmwrap(f"fmov {dst},{src}")

    def zero_freg(self, *, freg : freg_base, dt : adt) -> str:
        _ = dt # explicitly unused
        return self.asmwrap(f"fmov {freg},#0")

    def load_greg(self, *, areg : greg_base, offset : int, dst : greg_base) -> str:
        return self.asmwrap(f"ldr {dst},[{areg},#{offset}]")

    def store_greg(self, *, areg : greg_base, offset : int, src : greg_base) -> str:
        return self.asmwrap(f"str {src},[{areg},#{offset}]")

    def mov_param_to_greg(self, *, param : str, dst : greg_base) -> str:
        return self.asmwrap(f"ldr {dst},%[{param}]")

    def mov_param_to_greg_shift(self, *, param : str,
                                dst : greg_base, bit_count : int) -> str:
        return self.asmwrap(f"ldr {dst},%[{param},LSL#{bit_count}]")

    def mov_greg_to_param(self, *, src : greg_base, param : str) -> str:
        return self.asmwrap(f"str {src},%[{param}]")

    def mov_greg_imm(self, *, reg : greg_base, imm : int) -> str:
        return self.asmwrap(f"mov {reg},#{imm}")

    def mul_greg_imm(self, *, src : greg_base, dst : greg_base, factor):
        assert src != dst
        #Gotta do 2 instructions for this
        asmblock  = self.mov_greg_imm(reg=dst, imm=factor)
        #asmblock += self.asmwrap(f"mul {dst},{src},{dst}")
        asmblock += self.asmwrap(f"mul {dst},{src},{dst}")
        return asmblock

    def mul_greg_greg(self, *, dst : greg_base,
                      reg1 : greg_base, reg2 : greg_base) -> str:
        return self.asmwrap(f"mul {dst},{reg1},{reg2}")

    def add_greg_imm(self, *, reg : greg_base, imm : int) -> str:
        return self.asmwrap(f"add {reg},{reg},#{imm}")

    def add_greg_greg(self, *, dst : greg_base,
                      reg1 : greg_base, reg2 : greg_base) -> str:
        return self.asmwrap(f"add {dst},{reg1},{reg2}")

    def sub_greg_greg(self, *, dst : greg_base,
                      reg1 : greg_base, reg2 : greg_base) -> str:
        return self.asmwrap(f"sub {dst},{reg1},{reg2}")

    def shift_greg_left(self, *, reg : greg_base, bit_count : int) -> str:
        return self.asmwrap(f"lsl {reg},{reg},#{bit_count}")

    def shift_greg_right(self, *, reg : greg_base, bit_count : int) -> str:
        return self.asmwrap(f"lsr {reg},{reg},#{bit_count}")

    def zero_greg(self, *, greg : greg_base) -> str:
        return self.mov_greg_imm(reg=greg, imm=0)

    def greg(self, reg_idx : int) -> greg_base:
        return aarch64_greg(reg_idx)

    def freg(self, reg_idx : int, dt : adt) -> freg_base:
        return aarch64_freg(reg_idx=reg_idx, dt=dt)

    @property
    def min_prefetch_offset(self):
        return 0

    @property
    def max_prefetch_offset(self):
        return 32760

    def min_fload_immoff(self, dt : adt):
        return 0

    def max_fload_immoff(self, dt : adt):
        return 4095

    def load_scalar_immoff(self, *, areg : greg_base,
                           offset : int, freg : freg_base,
                           dt : adt) -> str:
        return self.asmwrap(f"ldr {freg},[{areg},#{offset}]")

    def store_scalar_immoff(self, *, areg : greg_base,
                            offset : int, freg : freg_base,
                            dt : adt) -> str:
        return self.asmwrap(f"str {freg},[{areg},#{offset}]")

    def load_freg(self, *, areg : greg_base, offset : int, dst: freg_base, dt : adt):
        return self.load_scalar_immoff(areg=areg, offset=offset, freg=dst, dt=dt)

    def store_freg(self, *, areg : greg_base, offset : int, src: freg_base, dt : adt):
        return self.store_scalar_immoff(areg=areg, offset=offset, freg=src, dt=dt)

    def prefetch_l1_immoff(self, *, areg : greg_base,
                         offset : int) -> str:
        return self.asmwrap(f"prfm pldl1strm,[{areg},#{offset}]")

    def load_pointer(self, *, areg : greg_base,
                     name : str) -> str:
        return self.asmwrap(f"ldr {areg},%[{name}]")
