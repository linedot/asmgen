from .noarch import asmgen
from ..registers import (
    reg_tracker,
    asm_data_type as adt,
    adt_triple,
    adt_size,
    asm_index_type as ait,
    data_reg,
    treg,vreg,freg,greg
)

from typing import TypeAlias

class riscv64_greg(greg):
    def __init__(self, reg_idx : int):
        self.reg_str = riscv64.greg_names[reg_idx];

    def __str__(self) -> str:
        return self.reg_str;

class riscv64_freg(freg):
    def __init__(self, reg_idx : int):
        self.reg_str = riscv64.freg_names[reg_idx];

    def __str__(self) -> str:
        return self.reg_str;

class riscv64(asmgen):

    greg_type : TypeAlias = greg
    freg_type : TypeAlias = freg

    # according to calling convention: temporaries, saved, function arguments,
    # leave the sp,gp, return address, etc... alone
    greg_names = [f't{i}' for i in range(7)] +\
                 [f's{i}' for i in range(1,12)] +\
                 [f'a{i}' for i in range(8)]

    fdt_suffixes = {
            adt.DOUBLE : "d",
            adt.SINGLE : "w",
            }

    fcdt_suffixes = {
            adt.DOUBLE : "d",
            adt.SINGLE : "s",
            }

    freg_names = [f'f{i}' for i in range(32)]

    def greg(self, reg_idx : int) -> greg_type:
        return riscv64_greg(reg_idx)

    def freg(self, reg_idx : int) -> freg_type:
        return riscv64_freg(reg_idx)

    @property
    def are_fregs_in_vregs(self) -> bool:
        return False

    def label(self, label : str) -> str:
        return self.asmwrap(f".{label}%=:")

    def jump(self, label : str) -> str:
        return self.asmwrap(f"j .{label}%=")

    def jzero(self, reg : greg_type, label : str) -> str:
        return self.asmwrap(f" beq {reg},zero,.{label}%=")

    def jfzero(self, freg1 : freg_type, freg2 : freg_type,
               greg : greg_type, label : str,
               dt : adt) -> str:
        dt_suf = self.fdt_suffixes[dt]
        cdt_suf = self.fcdt_suffixes[dt]
        asmblock  = self.asmwrap(f"fmv.{dt_suf}.x {freg2},zero")
        asmblock += self.asmwrap(f"feq.{cdt_suf} {greg},{freg1},{freg2}")
        asmblock += self.asmwrap(f"bnez {greg},.{label}%=")
        return asmblock

    def loopbegin(self, reg : greg_type, label : str) -> str:
        asmblock  = self.asmwrap(f".{label}%=:")
        asmblock += self.asmwrap(f"addi {reg},{reg},-1")

        return asmblock

    def loopbegin_nz(self, reg : greg_type, label : str, labelskip : str) -> str:
        asmblock  = self.asmwrap(f"beq {reg},zero,.{labelskip}%=")
        asmblock += self.asmwrap(f".{label}%=:")
        asmblock += self.asmwrap(f"addi {reg},{reg},-1")

        return asmblock

    def loopend(self, reg : greg_type, label : str):
        asmblock = f"\"bnez {reg}, .{label}%=\\n\\t\"\n"

        return asmblock

    @property
    def max_gregs(self):
        return len(self.greg_names)

    @property
    def max_fregs(self):
        return len(self.freg_names)

    def mov_freg(self, src : freg_type, dst : freg_type, dt : adt):
        dt_suf = self.fdt_suffixes[dt]
        return self.asmwrap(f"fmv.{dt_suf} {dst},{src}")

    def mov_greg(self, src : greg_type, dst : greg_type):
        # There is no mov instruction?
        return self.asmwrap(f"add {dst},{src},0")

    def mov_param_to_greg(self, param : str, dst : greg_type):
        return self.asmwrap(f"ld {dst},%[{param}]")

    def mov_param_to_greg_shift(self, param : str, dst : greg_type, bit_count : int):
        return self.asmwrap(f"slli {dst},%[{param}],{bit_count}")

    def mov_greg_to_param(self, src : greg_type, param : str):
        return self.asmwrap(f"sd {src},%[{param}]")

    def mov_greg_imm(self, reg : greg_type, imm : int):
        return self.asmwrap(f"li {reg},{imm}")

    def mul_greg_imm(self, src : greg_type, dst : greg_type, factor : int):
        #Gotta do 2 instructions for this
        asmblock  = self.mov_greg_imm(dst, factor)
        asmblock += self.asmwrap(f"mul {dst},{src},{dst}")
        return asmblock

    def add_greg_imm(self, reg : greg_type, offset : int):
        return self.asmwrap(f"add {reg},{reg},{offset}")

    def add_greg_greg(self, dst : greg_type, src : greg_type, reg2 : greg_type):
        return self.asmwrap(f"add {dst},{src},{reg2}")

    def sub_greg_greg(self, dst : greg_type, src : greg_type, reg2 : greg_type):
        return self.asmwrap(f"sub {dst},{src},{reg2}")

    def shift_greg_left(self, reg : greg_type, bit_count : int):
        return self.asmwrap(f"slli {reg},{reg},{bit_count}")

    def shift_greg_right(self, reg : greg_type, bit_count : int):
        return self.asmwrap(f"srli {reg},{reg},{bit_count}")

    def zero_greg(self, reg : greg_type):
        return self.mov_greg("zero", reg)

    def zero_freg(self, reg : greg_type):
        return self.asmwrap(f"fcvt.d.w {reg}, zero")

    @property
    def min_prefetch_offset(self):
        return 0

    @property
    def max_prefetch_offset(self):
        return 32760

    @property
    def min_load_immoff(self, dt : adt):
        return 0

    @property
    def max_load_immoff(self, dt : adt):
        return 0

    @property
    def min_fload_immoff(self, dt : adt):
        return -(1<<11)

    @property
    def max_fload_immoff(self, dt : adt):
        return (1<<11) -1

    def prefetch_l1_boff(self, areg : greg_type, offset : int):
        # Needs Zicbop
        return self.asmwrap(f"prefetch.r {offset}({areg})")

    def load_pointer(self, areg : greg_type, name : str):
        return self.asmwrap(f"ld {areg},%[{name}]")

    def load_scalar_immoff(self, areg : greg_type, offset : int, freg : freg_type,  dt : adt):
        dt_suf = self.fdt_suffixes[dt]
        return self.asmwrap(f"fl{dt_suf} {freg}, {offset}({areg})")
