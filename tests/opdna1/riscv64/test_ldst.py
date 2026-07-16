# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
RISCV64 +D/F opdna1 (ld/st) testsuite
"""
import unittest

from asmgen.asmblocks.op import opdna1_action, opdna1_modifier as mod
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.types.riscv64_types import riscv64_greg, riscv64_freg
from asmgen.asmblocks.riscv64_opdna1.riscv64_opdna1_base import riscv64_opdna1


def asmwrap(s: str) -> str:
    """
    Dummy asmwrap function to avoid pulling in more dependencies
    """
    lines = s.split("\n")
    return "".join(f"{line}\n" for line in lines)

class test_riscv64_opdna1(unittest.TestCase):
    """
    RISCV64 +D/F opdna1 tests
    """
    def setUp(self):
        self.t1 = riscv64_greg(1)
        self.t2 = riscv64_greg(2)
        self.f0 = riscv64_freg(0)
        self.f1 = riscv64_freg(1)

        self.load = riscv64_opdna1(action=opdna1_action.LOAD,
                                   asmwrap=asmwrap)
        self.store = riscv64_opdna1(action=opdna1_action.STORE,
                                    asmwrap=asmwrap)

    def test_basic_scalar_loads(self):
        """ Test scalar integer and floating point loads """
        self.assertEqual(
            self.load(dregs=[self.f1], areg=self.t1, dt=adt.FP64, modifiers=set()),
            "fld f1, 0(t1)\n"
        )
        self.assertEqual(
            self.load(dregs=[self.f0], areg=self.t1, dt=adt.FP32, modifiers=set()),
            "flw f0, 0(t1)\n"
        )
        self.assertEqual(
            self.load(dregs=[self.t2], areg=self.t1, dt=adt.SINT8, modifiers=set()),
            "lb t2, 0(t1)\n"
        )
        self.assertEqual(
            self.load(dregs=[self.t2], areg=self.t1, dt=adt.UINT64, modifiers=set()),
            "ld t2, 0(t1)\n"
        )

    def test_scalar_ioffset(self):
        """ Test scalar loads and stores with immediate offsets """
        self.assertEqual(
            self.load(dregs=[self.t2], areg=self.t1, dt=adt.SINT32,
                      modifiers={mod.IOFFSET}, ioffset=16),
            "lw t2, 16(t1)\n"
        )
        self.assertEqual(
            self.store(dregs=[self.f1], areg=self.t1, dt=adt.FP64,
                       modifiers={mod.IOFFSET}, ioffset=-8),
            "fsd f1, -8(t1)\n"
        )

    def test_scalar_invalid_modifiers(self):
        """ Ensure scalar class rejects invalid vector/matrix modifiers """
        invalid_mods = [mod.TINDEX, mod.VINDEX, mod.GLANE, mod.ILANE, mod.POSTINC,
                        mod.TOFFSET, mod.VOFFSET, mod.ISTRIDE, mod.GSTRIDE, mod.STRUCT]
        for invalid_mod in invalid_mods:
            with self.subTest(modifier=invalid_mod):
                with self.assertRaisesRegex(
                        ValueError,
                        ("riscv64_opdna1 does not support these modifiers "
                        f"at all: {{{invalid_mod.name}}}")):
                    self.load(dregs=[self.t2], areg=self.t1, dt=adt.UINT64, modifiers={invalid_mod})

if __name__ == '__main__':
    unittest.main()
