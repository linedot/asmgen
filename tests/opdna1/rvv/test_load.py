# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
import unittest

from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.types.rvv_types import rvv_vreg
from asmgen.asmblocks.types.riscv64_types import riscv64_greg
from asmgen.asmblocks.operations import opdna1_modifier as mod
from asmgen.registers import asm_data_type as adt

class test_rvv_load(unittest.TestCase):

    def setUp(self):
        # Create standard registers for testing
        self.vs = [rvv_vreg(i) for i in range(8)]
        self.vid = rvv_vreg(8)
        
        self.a0 = riscv64_greg(0)
        self.s0 = riscv64_greg(1)

        self.rvv = rvv()


    def test_unit_stride(self):
        """
        Test basic unit-stride loads with different data types
        """
        self.assertEqual(
            self.rvv.load(dregs=self.vs[:1],
                      areg=self.a0,
                      dt=adt.FP64,
                      modifiers={}),
            "vle64.v v0, (t0)"
        )
        self.assertEqual(
            self.rvv.load(dregs=self.vs[:1],
                      areg=self.a0,
                      dt=adt.FP32,
                      modifiers={}),
            "vle32.v v0, (t0)"
        )
        self.assertEqual(
            self.rvv.load(dregs=self.vs[:1],
                      areg=self.a0,
                      dt=adt.FP16,
                      modifiers={}),
            "vle16.v v0, (t0)"
        )

    def test_strided_load(self):
        """
        Test strided load with GSTRIDE modifier
        """
        self.assertEqual(
            self.rvv.load(dregs=self.vs[:1],
                      areg=self.a0, streg=self.s0,
                      dt=adt.FP64,
                      modifiers={mod.GSTRIDE}),
            "vlse64.v v0, (t0), t1"
        )

    def test_indexed_load(self):
        """
        Test indexed load with VINDEX modifier
        """
        self.assertEqual(
            self.rvv.load(dregs=self.vs[:1],
                      areg=self.a0, vidxreg=self.vid,
                      dt=adt.FP64,
                      modifiers={mod.VINDEX}),
            "vluxei64.v v0, (t0), v8"
        )

    def test_segmented_loads_lmul_1(self):
        """
        Test segmented loads requiring consecutive registers (LMUL=1)
        """
        self.assertEqual(
            self.rvv.load(dregs=self.vs[:2],
                      areg=self.a0, nstructs=2,
                      dt=adt.FP64,
                      modifiers={mod.STRUCT}),
            "vlseg2e64.v v0, (t0)"
        )
        self.assertEqual(
            self.rvv.load(dregs=self.vs[:3],
                      areg=self.a0,
                      streg=self.s0,
                      nstructs=3,
                      dt=adt.FP64,
                      modifiers={mod.STRUCT, mod.GSTRIDE}),
            "vlsseg3e64.v v0, (t0), t1"
        )
        self.assertEqual(
            self.rvv.load(dregs=self.vs[:4],
                      areg=self.a0,
                      vidxreg=self.vid,
                      nstructs=4,
                      dt=adt.FP64,
                      modifiers={mod.STRUCT, mod.VINDEX}),
            "vluxseg4ei64.v v0, (t0), v8"
        )

    def test_segmented_loads_lmul_2(self):
        """
        Test segmented loads with LMUL=2 requiring gaps between
        register groups
        """
        self.rvv.set_parameter("LMUL",2)
        
        # dregs: v0, v2 (valid for LMUL=2)
        dregs_m2 = [self.vs[0], self.vs[2]]
        
        self.assertEqual(
            self.rvv.load(dregs=dregs_m2,
                          areg=self.a0, nstructs=2,
                          dt=adt.FP64,
                          modifiers={mod.STRUCT}),
            "vlseg2e64.v v0, (t0)"
        )


    def test_lmul_sequence_violation(self):
        """
        Ensure exception is raised if register offsets don't match LMUL
        """
        self.rvv.set_parameter("LMUL",2)
        
        # dregs: v0, v1 (invalid for LMUL=2, should be v0, v2)
        dregs_invalid = [self.vs[0], self.vs[1]]
        
        with self.assertRaisesRegex(
                ValueError,
                "Segmented registers must be consecutive"):
            self.rvv.load(dregs=dregs_invalid,
                          areg=self.a0,
                          nstructs=2,
                          dt=adt.FP64,
                          modifiers={mod.STRUCT})

    def test_missing_required_kwargs(self):
        """
        Test missing keyword arguments for specific modifiers
        """
        with self.assertRaisesRegex(ValueError, "Missing parameter: streg"):
            self.rvv.load(dregs=self.vs[:1],
                          areg=self.a0,
                          dt=adt.FP64,
                          modifiers={mod.GSTRIDE})

        with self.assertRaisesRegex(ValueError, "Missing parameter: vidxreg"):
            self.rvv.load(dregs=self.vs[:1],
                          areg=self.a0,
                          dt=adt.FP64,
                          modifiers={mod.VINDEX})

        with self.assertRaisesRegex(ValueError, "Missing parameter: nstructs"):
            self.rvv.load(dregs=self.vs[:2],
                          areg=self.a0,
                          dt=adt.FP64,
                          modifiers={mod.STRUCT})

    def test_nstructs_mismatch(self):
        """
        Test when nstructs does not match the actual length of dregs
        provided
        """
        with self.assertRaisesRegex(ValueError,
                                    "3 nstructs specified but only 2 dregs given"):
            self.rvv.load(dregs=self.vs[:2],
                          areg=self.a0, nstructs=3,
                          dt=adt.FP64, modifiers={mod.STRUCT})

    def test_mutually_exclusive_modifiers(self):
        """
        Test that GSTRIDE and VINDEX cannot be used together
        """
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.rvv.load(dregs=self.vs[:1],
                          areg=self.a0,
                          streg=self.s0,
                          vidxreg=self.vid,
                          dt=adt.FP64,
                          modifiers={mod.GSTRIDE, mod.VINDEX})

    def test_unsupported_modifiers(self):
        """
        Test that unsupported modifiers correctly raise ValueError
        """
        unsupported = [
            mod.TINDEX, mod.GLANE, mod.ILANE, 
            mod.POSTINC, mod.TOFFSET, mod.VOFFSET, 
            mod.IOFFSET, mod.ISTRIDE
        ]
        
        for m in unsupported:
            with self.subTest(modifier=m):
                with self.assertRaises(ValueError):
                    self.rvv.load(dregs=self.vs[:1],
                                  areg=self.a0,
                                  dt=adt.FP64,
                                  modifiers={m})

    def test_invalid_register_types(self):
        """
        Ensure dregs and aregs reject improper types
        """
        # Bad dreg (passing a greg where a vreg is expected)
        with self.assertRaisesRegex(ValueError, "All dregs must be vregs"):
            self.rvv.load(dregs=[self.a0], areg=self.a0, dt=adt.FP64, modifiers={})
            
        # Bad areg (passing a vreg where a greg is expected)
        with self.assertRaisesRegex(ValueError, "is not a riscv64_greg"):
            self.rvv.load(dregs=self.vs[:1], areg=self.vs[0], dt=adt.FP64, modifiers={})

if __name__ == '__main__':
    unittest.main()
