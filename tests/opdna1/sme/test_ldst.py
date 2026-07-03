import unittest

from asmgen.asmblocks.operations import opdna1_action, opdna1_modifier as mod
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.types.aarch64_types import aarch64_greg, aarch64_freg
from asmgen.asmblocks.types.sve_types import sve_vreg,sve_preg
from asmgen.asmblocks.types.sme_types import sme_treg
from asmgen.asmblocks.sme_opdna1 import sme_load,sme_store

def asmwrap(s: str) -> str:
    return f"{s}\n"

class test_sme_opdna1(unittest.TestCase):
    def setUp(self):
        self.x0 = aarch64_greg(0)
        self.x1 = aarch64_greg(1)
        
        # GP registers for slice indices (usually w12-w15)
        self.w12 = aarch64_greg(12) 
        
        self.z0 = sve_vreg(0)
        self.z8 = sve_vreg(8)
        
        self.za0 = sme_treg(0, dt=adt.FP64)
        
        self.pn8 = sve_preg(8, is_pn=True)

        self.load = sme_load(asmwrap=asmwrap)
        self.store = sme_store(asmwrap=asmwrap)

    def test_routing_to_sve(self):
        """ Ensure normal Z register loads fall back to SVE """
        self.assertEqual(
            self.load(dregs=[self.z0], areg=self.x0, dt=adt.FP32, modifiers={}),
            "ld1w {z0.s}, p0/z, [x0]\n"
        )

    def test_sme_tile_slice_row(self):
        """ Test ZA tile horizontal (row) slice """
        self.assertEqual(
            self.load(dregs=[self.za0], areg=self.x0, dt=adt.FP64, 
                      modifiers={mod.ROW}, rowreg=self.w12, immrow=0),
            "ld1d {za0h.d[w12, 0]}, p0/z, [x0]\n"
        )

    def test_sme_tile_slice_col_with_goffset(self):
        """ Test ZA tile vertical (col) slice with scalar offset """
        self.assertEqual(
            self.store(dregs=[sme_treg(0,dt=adt.FP32)], areg=self.x0, dt=adt.FP32, 
                       modifiers={mod.COL, mod.GOFFSET}, colreg=self.w12, offreg=self.x1, immcol=1),
            "st1w {za0v.s[w12, 1]}, p0, [x0, x1, lsl #2]\n"
        )

    def test_sme_tile_missing_kwargs(self):
        """ Ensure missing rowreg/colreg throws an error """
        with self.assertRaisesRegex(ValueError, "Missing one of these parameters: rowreg"):
            self.load(dregs=[self.za0], areg=self.x0, dt=adt.FP64, modifiers={mod.ROW})

    def test_sme2_nontemporal_strided(self):
        """ Test SME2 LDNT1D with multiple strided registers and MUL VL """
        # Non-temporal requires mod.NT. We pass two Z registers to simulate strided pairs.
        self.assertEqual(
            self.load(dregs=[self.z0, self.z8], areg=self.x0, dt=adt.FP64, 
                      modifiers={mod.NT, mod.VOFFSET}, voffset=2, preg=self.pn8),
            "ldnt1d {z0.d, z8.d}, pn8/z, [x0, #2, MUL VL]\n"
        )

    def test_sme2_nontemporal_store(self):
        """ Test SME2 STNT1W (32-bit) with scalar index """
        self.assertEqual(
            self.store(dregs=[self.z0], areg=self.x0, dt=adt.FP32, 
                       modifiers={mod.NT, mod.GOFFSET}, offreg=self.x1, preg=self.pn8),
            "stnt1w {z0.s}, pn8, [x0, x1, lsl #2]\n"
        )

if __name__ == '__main__':
    unittest.main()
