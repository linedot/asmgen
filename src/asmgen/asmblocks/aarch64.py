from asmgen.asmblocks.noarch import asmgen
from asmgen.asmblocks.noarch import asm_data_type
from asmgen.asmblocks.noarch import greg, freg, vreg

import sys
if not sys.version_info >= (3, 10):
    from typing_extensions import TypeAlias
else:
    from typing import TypeAlias

class aarch64_greg(greg):
    def __init__(self, reg_idx : int):
        self.reg_str = f"x{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str

class aarch64_freg(freg):
    def __init__(self, reg_idx : int):
        # TODO: Other data types
        self.reg_str = f"d{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str

class aarch64(asmgen):

    greg_type : TypeAlias = greg
    freg_type : TypeAlias = freg
    vreg_type : TypeAlias = vreg

    dt_greg_pfx = {
            asm_data_type.DOUBLE : "x",
            asm_data_type.SINGLE : "w",
            }


    @property
    def are_fregs_in_vregs(self) -> bool:
        return True

    def label(self, label : str) -> str:
        return self.asmwrap(f".{label}%=:")

    def jump(self, label : str) -> str:
        return self.asmwrap(f"b .{label}%=")

    def loopbegin(self, reg : greg_type,
                  label : str) -> str:
        asmblock  = self.asmwrap(f".{label}%=:")
        asmblock += self.asmwrap(f"sub {reg},{reg},1")
        return asmblock

    def loopbegin_nz(self, reg : greg_type, 
                     label : str,
                     labelskip : str) -> str:
        asmblock  = self.asmwrap(f"cmp {reg},0")
        asmblock += self.asmwrap(f"b.eq .{labelskip}%=")
        asmblock += self.asmwrap(f".{label}%=:")
        asmblock += self.asmwrap(f"sub {reg},{reg},1")
        return asmblock

    def loopend(self, reg : greg_type,
                label : str) -> str:
        asmblock  = self.asmwrap(f"cmp {reg},0")
        asmblock += self.asmwrap(f"b.ne .{label}%=")
        return asmblock

    def jzero(self, reg : greg_type,
              label : str) -> str:
        asmblock  = self.asmwrap(f"cmp {reg},0")
        asmblock += self.asmwrap(f"b.eq .{label}%=")
        return asmblock

    def jfzero(self, freg1 : freg_type, freg2 : freg_type,
               greg : greg_type, label : str,
               datatype : asm_data_type):
        asmblock  = self.asmwrap(f"fcmp {freg1},#0.0")
        asmblock += self.asmwrap(f"b.eq .{label}%=")
        return asmblock

    @property
    def max_gregs(self):
        return 32

    @property
    def max_fregs(self):
        return 32

    def mov_greg(self, src : greg_type, dst : greg_type) -> str:
        return self.asmwrap(f"mov {dst},{src}")

    def mov_freg(self, src : freg_type, dst : freg_type,
                 datatype : asm_data_type) -> str:
        return self.asmwrap(f"fmov {dst},{src}")

    def zero_freg(self, reg : freg_type) -> str:
        return self.asmwrap(f"fmov {reg},#0")

    def fma_vf(self, avreg : vreg_type, bfreg : freg_type, cvreg : vreg_type,
               datatype : asm_data_type) -> str:
        raise NotImplementedError("NEON/SVE doesn't have vector x scalar FMA instruction")

    def fmul_vf(self, avreg : vreg_type, bfreg : freg_type, cvreg : vreg_type,
                datatype : asm_data_type) -> str:
        raise NotImplementedError("NEON/SVE doesn't have vector x scalar FMUL instruction")

    def mov_param_to_greg(self, param : str, dst : greg_type) -> str:
        return self.asmwrap(f"ldr {dst},%[{param}]")

    def mov_param_to_greg_shift(self, param : str, 
                                dst : greg_type, offset : int) -> str:
        return self.asmwrap(f"ldr {dst},%[{param},LSL#{offset}]")

    def mov_greg_to_param(self, src : greg_type, param : str) -> str:
        return self.asmwrap(f"str {src},%[{param}]")

    def mov_greg_imm(self, reg : greg_type, imm : int) -> str:
        return self.asmwrap(f"mov {reg},#{imm}")

    def mul_greg_imm(self, src, dst, offset):
        #Gotta do 2 instructions for this
        asmblock  = self.mov_greg_imm(dst, offset)
        asmblock += self.asmwrap(f"mul {dst},{src},{dst}")
        return asmblock

    def add_greg_imm(self, reg : greg_type, imm : int) -> str:
        return self.asmwrap(f"add {reg},{reg},#{imm}")

    def add_greg_greg(self, dst : greg_type,
                      reg1 : greg_type, reg2 : greg_type) -> str:
        return self.asmwrap(f"add {dst},{reg1},{reg2}")

    @property
    def has_add_greg_voff(self):
        return True

    def shift_greg_left(self, reg : greg_type, offset : int) -> str:
        return self.asmwrap(f"lsl {reg},{reg},#{offset}")

    def shift_greg_right(self, reg : greg_type, offset : int) -> str:
        return self.asmwrap(f"lsr {reg},{reg},#{offset}")

    def zero_greg(self, reg : greg_type) -> str:
        return self.mov_greg_imm(reg,0)

    def greg(self, reg_idx : int) -> greg:
        return aarch64_greg(reg_idx)

    def freg(self, reg_idx : int) -> freg:
        # TODO: different data types
        return aarch64_freg(reg_idx)
    
    @property
    def min_prefetch_offset(self):
        return 0

    @property
    def max_prefetch_offset(self):
        return 32760

    @property
    def max_fload_immoff(self):
        return 4095

    def load_scalar_immoff(self, areg : greg_type,
                           offset : int, freg : freg_type,
                           datatype : asm_data_type) -> str:
        return self.asmwrap(f"ldr {freg},[{areg},#{offset}]")

    def prefetch_l1_boff(self, areg : greg_type,
                         offset : int) -> str:
        return self.asmwrap(f"prfm pldl1strm,[{areg},#{offset}]")

    def load_pointer(self, areg : greg_type, 
                     name : str) -> str:
        return self.asmwrap(f"ldr {areg},%[{name}]")

