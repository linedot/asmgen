# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Tests SME loads/stores
"""
import unittest

from asmgen.asmblocks.op import opdna1_modifier as mod
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.types.aarch64_types import aarch64_greg
from asmgen.asmblocks.types.sve_types import sve_vreg,sve_preg
from asmgen.asmblocks.types.sme_types import sme_treg
from asmgen.asmblocks.sme_opdna1 import sme_load,sme_store

def asmwrap(s: str) -> str:
    """
    Dummy asmwrap
    """
    return f"{s}\n"

# fine for testing
# pylint: disable-next=too-many-instance-attributes
class test_sme_opdna1(unittest.TestCase):
    """
    Testsuite for SME opdna1 operations
    """
    def setUp(self):
        self.x0 = aarch64_greg(0)
        self.x1 = aarch64_greg(1)

        # GP registers for slice indices (usually w12-w15)
        self.w12 = aarch64_greg(12)

        # GP regs for testing col/row restriction
        self.w5  = aarch64_greg(5)
        self.w17  = aarch64_greg(17)

        self.z0 = sve_vreg(0)
        self.z8 = sve_vreg(8)

        self.za0 = sme_treg(0, dt=adt.FP64)

        self.p0  = sve_preg(0)
        self.pn8 = sve_preg(8, is_pn=True)

        self.load = sme_load(asmwrap=asmwrap)
        self.store = sme_store(asmwrap=asmwrap)

    def test_routing_to_sve(self):
        """ Ensure normal Z register loads fall back to SVE """
        self.assertEqual(
            self.load(dregs=[self.z0], areg=self.x0,
                      amreg=self.p0,
                      dt=adt.FP32,
                      modifiers={mod.MASK}),
            "ld1w {z0.s}, p0/z, [x0]\n"
        )

    def test_sme_tile_slice_row(self):
        """ Test ZA tile horizontal (row) slice """
        self.assertEqual(
            self.load(dregs=[self.za0], areg=self.x0,
                      amreg=self.p0,
                      dt=adt.FP64,
                      modifiers={mod.ROW, mod.MASK},
                      rowreg=self.w12, immrow=0),
            "ld1d {za0h.d[w12, 0]}, p0/z, [x0]\n"
        )

    def test_sme_tile_slice_col_with_goffset(self):
        """ Test ZA tile vertical (col) slice with scalar offset """
        self.assertEqual(
            self.store(dregs=[sme_treg(0,dt=adt.FP32)],
                       areg=self.x0, dt=adt.FP32,
                       amreg=self.p0,
                       modifiers={mod.COL, mod.GOFFSET, mod.MASK},
                       colreg=self.w12, offreg=self.x1, immcol=1),
            "st1w {za0v.s[w12, 1]}, p0, [x0, x1, lsl #2]\n"
        )

    def test_sme_tile_missing_kwargs(self):
        """ Ensure missing rowreg/colreg throws an error """
        with self.assertRaisesRegex(ValueError,
                                    "ROW modifier requires 'rowreg' parameter"):
            self.load(dregs=[self.za0], areg=self.x0, dt=adt.FP64, modifiers={mod.ROW})

    def test_sme_rowreg_index_too_small(self):
        """ Ensure too small rowreg/colreg index throws an error """
        with self.assertRaisesRegex(ValueError,
                                    "index of rowreg must be >= 12"):
            self.load(dregs=[self.za0],
                      areg=self.x0,
                      amreg=self.p0,
                      dt=adt.FP64,
                      modifiers={mod.ROW, mod.MASK},
                      rowreg=self.w5,
                      immrow=0)

    def test_sme_rowreg_index_too_big(self):
        """ Ensure too small rowreg/colreg index throws an error """
        with self.assertRaisesRegex(ValueError,
                                    "index of rowreg must be <= 15"):
            self.load(dregs=[self.za0],
                      areg=self.x0,
                      amreg=self.p0,
                      dt=adt.FP64,
                      modifiers={mod.ROW, mod.MASK},
                      rowreg=self.w17,
                      immrow=0)

    def test_sme2_nontemporal_strided(self):
        """ Test SME2 LDNT1D with multiple strided registers and MUL VL """
        # Non-temporal requires mod.NT. We pass two Z registers to simulate strided pairs.
        self.assertEqual(
            self.load(dregs=[self.z0, self.z8], areg=self.x0,
                      amreg=self.pn8, dt=adt.FP64,
                      modifiers={mod.NT, mod.STRUCT, mod.MASK, mod.VOFFSET},
                      nstructs=2, voffset=2),
            "ldnt1d {z0.d, z8.d}, pn8/z, [x0, #2, MUL VL]\n"
        )

    def test_sme2_nontemporal_strided_store_spluss(self):
        """ Test SME2 STNT1W (32-bit) with scalar index """
        self.assertEqual(
            self.store(dregs=[self.z0, self.z8], areg=self.x0,
                       amreg = self.pn8, dt=adt.FP32,
                       modifiers={mod.NT, mod.STRUCT, mod.MASK, mod.GOFFSET},
                       nstructs=2, offreg=self.x1),
            "stnt1w {z0.s, z8.s}, pn8, [x0, x1, lsl #2]\n"
        )

    def test_sme2_nontemporal_strided_pn9(self):
        """ Ensure amreg is correctly parsed by using pn9 """
        pn9 = sve_preg(9, is_pn=True)
        self.assertEqual(
            self.load(dregs=[self.z0, self.z8], areg=self.x0,
                      amreg=pn9, dt=adt.FP64,
                      modifiers={mod.NT, mod.STRUCT, mod.MASK, mod.VOFFSET},
                      nstructs=2, voffset=2),
            "ldnt1d {z0.d, z8.d}, pn9/z, [x0, #2, MUL VL]\n"
        )

    def test_sme_constraint_rejections(self):
        """ Test declarative signature rejections for SME constraints """

        # 1. Tile slice using a PN predicate (must be p0-p7)
        with self.subTest(error="Tile slice with PN predicate"):
            with self.assertRaisesRegex(ValueError, "index of amreg must be <= 7"):
                self.load(dregs=[self.za0], areg=self.x0, amreg=self.pn8, dt=adt.FP64,
                          modifiers={mod.ROW, mod.MASK}, rowreg=self.w12, immrow=0)

        # 2. NT strided load using a standard P predicate (must be pn8-pn15)
        with self.subTest(error="NT load with P predicate"):
            with self.assertRaisesRegex(ValueError, "index of amreg must be >= 8"):
                self.load(dregs=[self.z0, self.z8], areg=self.x0, amreg=self.p0, dt=adt.FP64,
                          modifiers={mod.NT, mod.STRUCT, mod.MASK}, nstructs=2)

        # 3. NT strided load with non-strided registers (e.g. z0, z1 instead of z0, z8)
        with self.subTest(error="NT registers not strided correctly"):
            with self.assertRaisesRegex(ValueError,
                                        r"index of bdreg must be index of adreg plus 8 modulo 32"):
                self.load(dregs=[self.z0, sve_vreg(1)], areg=self.x0, amreg=self.pn8, dt=adt.FP64,
                          modifiers={mod.NT, mod.STRUCT, mod.MASK}, nstructs=2)

        # 4. NT strided load with invalid starting register (e.g. z15)
        with self.subTest(error="NT invalid starting register"):
            with self.assertRaisesRegex(ValueError, r"index of adreg must be one of"):
                self.load(dregs=[sve_vreg(15), sve_vreg(23)], areg=self.x0,
                          amreg=self.pn8, dt=adt.FP64,
                          modifiers={mod.NT, mod.STRUCT, mod.MASK}, nstructs=2)

        # 5. Tile slice immediate out of bounds (FP64 max is 1)
        with self.subTest(error="Tile slice immediate bounds"):
            with self.assertRaisesRegex(ValueError, "value of immrow must be <= 1"):
                self.load(dregs=[self.za0], areg=self.x0, amreg=self.p0, dt=adt.FP64,
                          modifiers={mod.ROW, mod.MASK}, rowreg=self.w12, immrow=2)

if __name__ == '__main__':
    unittest.main()
