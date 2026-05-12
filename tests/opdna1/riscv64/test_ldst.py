# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
import unittest

from asmgen.asmblocks.operations import opdna1_action, opdna1_modifier as mod
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.types.riscv64_types import riscv64_greg, riscv64_freg
from asmgen.asmblocks.riscv64_opdna1.riscv64_opdna1_base import riscv64_opdna1

class test_riscv64_opdna1(unittest.TestCase):
    def setUp(self):
        self.t1 = riscv64_greg(1)
        self.t2 = riscv64_greg(2)
        self.f0 = riscv64_freg(0)
        self.f1 = riscv64_freg(1)
        
        self.load = riscv64_opdna1(action=opdna1_action.LOAD)
        self.store = riscv64_opdna1(action=opdna1_action.STORE)

    def test_basic_scalar_loads(self):
        """ Test scalar integer and floating point loads """
        self.assertEqual(
            self.load(dregs=[self.f1], areg=self.t1, dt=adt.FP64, modifiers={}),
            "fld f1, 0(t1)"
        )
        self.assertEqual(
            self.load(dregs=[self.f0], areg=self.t1, dt=adt.FP32, modifiers={}),
            "flw f0, 0(t1)"
        )
        self.assertEqual(
            self.load(dregs=[self.t2], areg=self.t1, dt=adt.SINT8, modifiers={}),
            "lb t2, 0(t1)"
        )
        self.assertEqual(
            self.load(dregs=[self.t2], areg=self.t1, dt=adt.UINT64, modifiers={}),
            "ld t2, 0(t1)"
        )

    def test_scalar_ioffset(self):
        """ Test scalar loads and stores with immediate offsets """
        self.assertEqual(
            self.load(dregs=[self.t2], areg=self.t1, dt=adt.SINT32, 
                      modifiers={mod.IOFFSET}, ioffset=16),
            "lw t2, 16(t1)"
        )
        self.assertEqual(
            self.store(dregs=[self.f1], areg=self.t1, dt=adt.FP64, 
                       modifiers={mod.IOFFSET}, ioffset=-8),
            "fsd f1, -8(t1)"
        )

    def test_scalar_invalid_modifiers(self):
        """ Ensure scalar class rejects invalid vector/matrix modifiers """
        invalid_mods = [mod.TINDEX, mod.VINDEX, mod.GLANE, mod.ILANE, mod.POSTINC, 
                        mod.TOFFSET, mod.VOFFSET, mod.ISTRIDE, mod.GSTRIDE, mod.STRUCT]
        for invalid_mod in invalid_mods:
            with self.subTest(modifier=invalid_mod):
                with self.assertRaisesRegex(ValueError, r"RISC-V \+D/F has no"):
                    self.load(dregs=[self.t2], areg=self.t1, dt=adt.UINT64, modifiers={invalid_mod})

if __name__ == '__main__':
    unittest.main()
