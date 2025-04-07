"""
RISC-V 64bit asm generator and related types
"""

from ..registers import (
    asm_data_type as adt,
    greg_base, freg_base
)

from .noarch import asmgen

from .types.riscv64_types import riscv64_freg, riscv64_greg


# pylint: disable=too-many-public-methods

class riscv64(asmgen):
    """
    RISC-V 64bit asmgen implementation
    """

    fdt_suffixes = {
            adt.DOUBLE : "d",
            adt.SINGLE : "w",
            }

    fcdt_suffixes = {
            adt.DOUBLE : "d",
            adt.SINGLE : "s",
            }

    def greg(self, reg_idx : int) -> greg_base:
        return riscv64_greg(reg_idx)

    def freg(self, reg_idx : int, dt : adt) -> freg_base:
        _ = dt # explicitly unused
        return riscv64_freg(reg_idx)

    @property
    def are_fregs_in_vregs(self) -> bool:
        return False

    def label(self, *, label : str) -> str:
        return self.asmwrap(f".{label}%=:")

    def jump(self, *, label : str) -> str:
        return self.asmwrap(f"j .{label}%=")

    def jzero(self, *, reg : greg_base, label : str) -> str:
        return self.asmwrap(f" beq {reg},zero,.{label}%=")

    def jfzero(self, *, freg1 : freg_base, freg2 : freg_base,
               greg : greg_base, label : str,
               dt : adt) -> str:
        dt_suf = self.fdt_suffixes[dt]
        cdt_suf = self.fcdt_suffixes[dt]
        asmblock  = self.asmwrap(f"fmv.{dt_suf}.x {freg2},zero")
        asmblock += self.asmwrap(f"feq.{cdt_suf} {greg},{freg1},{freg2}")
        asmblock += self.asmwrap(f"bnez {greg},.{label}%=")
        return asmblock

    def loopbegin(self, *, reg : greg_base, label : str) -> str:
        asmblock  = self.asmwrap(f".{label}%=:")
        asmblock += self.asmwrap(f"addi {reg},{reg},-1")

        return asmblock

    def loopbegin_nz(self, *, reg : greg_base, label : str, labelskip : str) -> str:
        asmblock  = self.asmwrap(f"beq {reg},zero,.{labelskip}%=")
        asmblock += self.asmwrap(f".{label}%=:")
        asmblock += self.asmwrap(f"addi {reg},{reg},-1")

        return asmblock

    def loopend(self, *, reg : greg_base, label : str) -> str:
        asmblock = f"\"bnez {reg}, .{label}%=\\n\\t\"\n"

        return asmblock

    @property
    def max_gregs(self) -> int:
        return len(riscv64_greg.names)

    @property
    def max_fregs(self) -> int:
        return len(riscv64_freg.names)

    def mov_freg(self, *, src : freg_base, dst : freg_base, dt : adt) ->  str:
        dt_suf = self.fdt_suffixes[dt]
        return self.asmwrap(f"fmv.{dt_suf} {dst},{src}")

    def mov_greg(self, *, src : greg_base, dst : greg_base) -> str:
        # There is no mov instruction?
        return self.asmwrap(f"add {dst},{src},0")

    def mov_param_to_greg(self, *, param : str, dst : greg_base) -> str:
        return self.asmwrap(f"ld {dst},%[{param}]")

    def mov_param_to_greg_shift(self, *, param : str, dst : greg_base, bit_count : int) -> str:
        return self.asmwrap(f"slli {dst},%[{param}],{bit_count}")

    def mov_greg_to_param(self, *, src : greg_base, param : str) -> str:
        return self.asmwrap(f"sd {src},%[{param}]")

    def mov_greg_imm(self, *, reg : greg_base, imm : int) -> str:
        return self.asmwrap(f"li {reg},{imm}")

    def mul_greg_imm(self, *, src : greg_base, dst : greg_base, factor : int) -> str:
        assert src != dst
        #Gotta do 2 instructions for this
        asmblock  = self.mov_greg_imm(reg=dst, imm=factor)
        asmblock += self.asmwrap(f"mul {dst},{src},{dst}")
        return asmblock

    def add_greg_imm(self, *, reg : greg_base, imm : int) -> str:
        return self.asmwrap(f"add {reg},{reg},{imm}")

    def add_greg_greg(self, *, dst : greg_base, reg1 : greg_base, reg2 : greg_base) -> str:
        return self.asmwrap(f"add {dst},{reg1},{reg2}")

    def sub_greg_greg(self, *, dst : greg_base, reg1 : greg_base, reg2 : greg_base) -> str:
        return self.asmwrap(f"sub {dst},{reg1},{reg2}")

    def shift_greg_left(self, *, reg : greg_base, bit_count : int) -> str:
        return self.asmwrap(f"slli {reg},{reg},{bit_count}")

    def shift_greg_right(self, *, reg : greg_base, bit_count : int) -> str:
        return self.asmwrap(f"srli {reg},{reg},{bit_count}")

    def zero_greg(self, *, greg : greg_base) -> str:
        return self.asmwrap(f"add {greg}, zero, 0")

    def zero_freg(self, *, freg : freg_base, dt : adt) -> str:
        dt_suf = self.fdt_suffixes[dt]
        return self.asmwrap(f"fcvt.{dt_suf}.w {freg}, zero")

    @property
    def min_prefetch_offset(self):
        return 0

    @property
    def max_prefetch_offset(self):
        return 32760

    def min_load_immoff(self, dt : adt):
        _ = dt # explicitly unused
        return 0

    def max_load_immoff(self, dt : adt):
        _ = dt # explicitly unused
        return 0

    def min_fload_immoff(self, dt : adt):
        _ = dt # explicitly unused
        return -(1<<11)

    def max_fload_immoff(self, dt : adt):
        _ = dt # explicitly unused
        return (1<<11) -1

    def prefetch_l1_boff(self, *, areg : greg_base, offset : int) -> str:
        # Needs Zicbop
        return self.asmwrap(f"prefetch.r {offset}({areg})")

    def load_pointer(self, *, areg : greg_base, name : str) -> str:
        return self.asmwrap(f"ld {areg},%[{name}]")

    def load_scalar_immoff(self, *, areg : greg_base, offset : int,
                           freg : freg_base,  dt : adt) -> str:
        dt_suf = self.fdt_suffixes[dt]
        return self.asmwrap(f"fl{dt_suf} {freg}, {offset}({areg})")
