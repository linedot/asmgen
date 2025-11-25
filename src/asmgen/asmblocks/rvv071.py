# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RISC-V RVV 0.7.1 asm generator and related types
"""

from .rvv import rvv

from ..registers import asm_data_type as adt, adt_size, greg_base

# pylint: disable=too-many-public-methods
class rvv071(rvv):
    """
    RISC-V RVV 0.7.1 asmgen implementation
    """

    dt_suffixes = {
            adt.DOUBLE  : "e",
            adt.SINGLE  : "w",
            adt.HALF    : "h",
            adt.FP8E4M3 : "b",
            adt.FP8E5M2 : "b",
            }

    def simd_size_to_greg(self, *, reg : greg_base,
                          dt : adt) -> str:
        esfx = adt_size(dt)*8
        return self.asmwrap(f"vsetvli {reg}, zero, e{esfx}, m{self.lmul}")

    @property
    def c_simd_size_function(self):
        pre_oi = self.output_inline
        self.set_output_inline(yesno=True)
        result  = "size_t get_simd_size() {\n"
        result += "    size_t byte_size = 0;\n"
        result += "    __asm__ volatile(\n"
        result += "        "+self.asmwrap(f"vsetvli %[byte_size], zero, e8, m{self.lmul}")
        result += "    : [byte_size] \"=r\" (byte_size)\n"
        result += "    :\n"
        result += "    :\n"
        result += "    );\n"
        result += "    return byte_size;\n"
        result += "}"
        self.set_output_inline(yesno=pre_oi)
        return result

    def vsetvli(self, *, vlreg : greg_base, avlreg : greg_base, dt : adt) -> str:
        dt_size = 'e'+str(adt_size(dt)*8)
        return self.asmwrap(f"vsetvli {vlreg}, {avlreg}, {dt_size}, m1")

    def vsetvlmax(self, *, reg : greg_base, dt : adt) -> str:
        dt_size = 'e'+str(adt_size(dt)*8)
        return self.asmwrap(f"vsetvli {reg}, zero, {dt_size}, m1")
