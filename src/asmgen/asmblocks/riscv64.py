from asmgen.asmblocks.noarch import asmgen
from asmgen.asmblocks.noarch import asm_data_type
from asmgen.asmblocks.noarch import greg, freg

import sys
if not sys.version_info >= (3, 10):
    from typing_extensions import TypeAlias
else:
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
            asm_data_type.DOUBLE : "d",
            asm_data_type.SINGLE : "w",
            }

    fcdt_suffixes = {
            asm_data_type.DOUBLE : "d",
            asm_data_type.SINGLE : "s",
            }

    freg_names = [f'f{i}' for i in range(32)]

    def greg(self, reg_idx : int) -> greg_type:
        return riscv64_greg(reg_idx)

    def freg(self, reg_idx : int) -> freg_type:
        return riscv64_freg(reg_idx)

    @property
    def are_fregs_in_vregs(self):
        return False

    def label(self, label : str) -> str:
        return self.asmwrap(f".{label}:")

    def jump(self, label : str) -> str:
        return self.asmwrap(f"j .{label}")

    def jzero(self, reg : greg_type, label : str):
        return self.asmwrap(f" beq {reg},zero,.{label}")

    def jfzero(self, freg1 : freg_type, freg2 : freg_type,
               greg : greg_type, label : str,
               datatype : asm_data_type):
        dt_suf = self.fdt_suffixes[datatype]
        cdt_suf = self.fcdt_suffixes[datatype]
        asmblock  = self.asmwrap(f"fmv.{dt_suf}.x {freg2},zero")
        asmblock += self.asmwrap(f"feq.{cdt_suf} {greg},{freg1},{freg2}")
        asmblock += self.asmwrap(f"bnez {greg},.{label}")
        return asmblock

    def loopbegin(self, reg, label):
        asmblock  = self.asmwrap(f".{label}:")
        asmblock += self.asmwrap(f"addi {reg},{reg},-1")

        return asmblock

    def loopbegin_nz(self, reg, label, labelskip):
        asmblock  = self.asmwrap(f"beq {reg},zero,.{labelskip}")
        asmblock += self.asmwrap(f".{label}:")
        asmblock += self.asmwrap(f"addi {reg},{reg},-1")

        return asmblock

    def loopend(self, reg, label):
        asmblock = f"\"bnez {reg}, .{label}\\n\\t\"\n"

        return asmblock

    @property
    def max_gregs(self):
        return len(self.greg_names)

    @property
    def max_fregs(self):
        return len(self.freg_names)

    def mov_freg(self, src, dst, datatype : asm_data_type):
        dt_suf = self.fdt_suffixes[datatype]
        return self.asmwrap(f"fmv.{dt_suf} {dst},{src}")

    def mov_greg(self, src, dst):
        # There is no mov instruction?
        return self.asmwrap(f"add {dst},{src},0")

    def mov_param_to_greg(self, param, dst):
        return self.asmwrap(f"ld {dst},%[{param}]")

    def mov_param_to_greg_shift(self, param, dst, offset):
        return self.asmwrap(f"slli {dst},%[{param}],offset")

    def mov_greg_to_param(self, src, param):
        return self.asmwrap(f"sd {src},%[{param}]")

    def mov_greg_imm(self, reg, imm):
        return self.asmwrap(f"li {reg},{imm}")

    def add_greg_imm(self, reg, offset):
        return self.asmwrap(f"add {reg},{reg},{offset}")

    def add_greg_greg(self, dst, reg1, reg2):
        return self.asmwrap(f"add {dst},{reg1},{reg2}")

    def shift_greg_left(self, reg, offset):
        return self.asmwrap(f"slli {reg},{reg},{offset}")

    def shift_greg_right(self, reg, offset):
        return self.asmwrap(f"srli {reg},{reg},{offset}")

    def zero_greg(self, reg):
        return self.mov_greg("zero", reg)

    def zero_freg(self, reg):
        return self.asmwrap(f"fcvt.d.w {reg}, zero")

    @property
    def min_prefetch_offset(self):
        return 0

    @property
    def max_prefetch_offset(self):
        return 32760

    def min_load_immoff(self,datatype):
        return 0

    def max_load_immoff(self,datatype):
        return 0

    def max_fload_immoff(self,datatype):
        return (1<<12) -1

    def prefetch_l1_boff(self, a, offset):
        # Needs Zicbop
        return self.asmwrap(f"prefetch.r {offset}({a})")

    def load_pointer(self, a, name):
        return self.asmwrap(f"ld {a},%[{name}]")

    def load_scalar_immoff(self, areg, offset, freg, datatype):
        assert isinstance(datatype, asm_data_type), f"Not an asm_data_type: {datatype}"
        dt_suf = self.fdt_suffixes[datatype]
        return self.asmwrap(f"fl{dt_suf} {freg}, {offset}({areg})")
