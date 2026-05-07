# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
import unittest

from asmgen.asmblocks.riscv64_opdna1.riscv64_store import riscv64_store
from asmgen.asmblocks.types.riscv64_types import riscv64_greg, riscv64_freg
from asmgen.asmblocks.types.rvv_types import rvv_vreg
from asmgen.asmblocks.operations import opdna1_modifier as mod
from asmgen.registers import asm_data_type as adt

class test_riscv64_store(unittest.TestCase):

    def setUp(self):
        self.t0 = riscv64_greg(0)
        self.t1 = riscv64_greg(1)
        
        # Floating point register
        self.ft0 = riscv64_freg(0)
        
        # Vector register (used to test invalid type handling)
        self.v0 = rvv_vreg(0)

        # Initialize base RISC-V store instruction
        self.store = riscv64_store()

    def test_gp_reg_stores_basic(self):
        """
        Test basic GP-register stores (zero offset) with different data types
        """
        self.assertEqual(
            self.store(dregs=[self.t0], 
                      areg=self.t1, 
                      dt=adt.SINT64, 
                      modifiers=set()),
            "sd t0, 0(t1)"
        )
        self.assertEqual(
            self.store(dregs=[self.t0], 
                      areg=self.t1, 
                      dt=adt.SINT32, 
                      modifiers=set()),
            "sw t0, 0(t1)"
        )
        self.assertEqual(
            self.store(dregs=[self.t0], 
                      areg=self.t1, 
                      dt=adt.SINT16, 
                      modifiers=set()),
            "sh t0, 0(t1)"
        )
        self.assertEqual(
            self.store(dregs=[self.t0], 
                      areg=self.t1, 
                      dt=adt.SINT8,
                      modifiers=set()),
            "sb t0, 0(t1)"
        )

    def test_fp_reg_stores_basic(self):
        """
        Test basic F-register floating-point stores
        """
        self.assertEqual(
            self.store(dregs=[self.ft0], 
                      areg=self.t1, 
                      dt=adt.FP64, 
                      modifiers=set()),
            "fsd f0, 0(t1)"
        )
        self.assertEqual(
            self.store(dregs=[self.ft0], 
                      areg=self.t1, 
                      dt=adt.FP32, 
                      modifiers=set()),
            "fsw f0, 0(t1)"
        )
        self.assertEqual(
            self.store(dregs=[self.ft0], 
                      areg=self.t1, 
                      dt=adt.FP16, 
                      modifiers=set()),
            "fsh f0, 0(t1)"
        )

    def test_immediate_offset(self):
        """
        Test stores with the IOFFSET modifier and specific immediate values
        """
        # GP-reg with positive offset
        self.assertEqual(
            self.store(dregs=[self.t0], 
                      areg=self.t1, 
                      dt=adt.SINT64, 
                      modifiers={mod.IOFFSET}, 
                      ioffset=32),
            "sd t0, 32(t1)"
        )
        # FP-reg with negative offset
        self.assertEqual(
            self.store(dregs=[self.ft0], 
                      areg=self.t1, 
                      dt=adt.FP64, 
                      modifiers={mod.IOFFSET}, 
                      ioffset=-16),
            "fsd f0, -16(t1)"
        )

    def test_missing_required_kwargs(self):
        """
        Test missing keyword arguments for the IOFFSET modifier
        """
        with self.assertRaisesRegex(ValueError, "Missing parameter: ioffset"):
            self.store(dregs=[self.t0], 
                      areg=self.t1, 
                      dt=adt.SINT64, 
                      modifiers={mod.IOFFSET})

    def test_invalid_dregs_count(self):
        """
        Ensure base stores strictly enforce a single destination register
        """
        with self.assertRaisesRegex(ValueError, "uses one and only one register"):
            self.store(dregs=[self.t0, self.t1], 
                      areg=self.t1, 
                      dt=adt.SINT64, 
                      modifiers=set())

    def test_invalid_register_types(self):
        """
        Ensure dregs and aregs reject improper register types
        """
        # Bad dreg (passing a vreg where a greg/freg is expected)
        with self.assertRaisesRegex(ValueError, "is neither a riscv64_freg nor a riscv64_greg"):
            self.store(dregs=[self.v0], 
                      areg=self.t1, 
                      dt=adt.SINT64, 
                      modifiers=set())
            
        # Bad areg (passing a vreg where a greg is expected)
        with self.assertRaisesRegex(ValueError, "is not a riscv64_greg"):
            self.store(dregs=[self.t0], 
                      areg=self.v0, 
                      dt=adt.SINT64, 
                      modifiers=set())

    def test_unsupported_modifiers(self):
        """
        Test that unsupported modifiers correctly raise ValueError
        """
        unsupported = [
            mod.TINDEX, mod.VINDEX, mod.GLANE, mod.ILANE, 
            mod.POSTINC, mod.TOFFSET, mod.VOFFSET, 
            mod.ISTRIDE, mod.GSTRIDE, mod.STRUCT
        ]
        
        for m in unsupported:
            with self.subTest(modifier=m):
                with self.assertRaises(ValueError):
                    self.store(dregs=[self.t0], 
                              areg=self.t1, 
                              dt=adt.SINT64, 
                              modifiers={m})

if __name__ == '__main__':
    unittest.main()
