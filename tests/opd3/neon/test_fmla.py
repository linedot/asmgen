"""
Tests NEON/ASIMD fma instruction code generation
"""
import unittest

from asmgen.asmblocks.neon import neon
from asmgen.registers import asm_data_type as adt

from asmgen.asmblocks.operations import modifier as mod

class test_neon_opd3(unittest.TestCase):
    """
    Tests NEON/ASIMD opd3 operations
    """
    def test_fmla(self):
        """
        Tests that the NEON/ASIMD generator generates correct FMA instructions
        """
        gen = neon()
        gen.set_output_inline(yesno=False)

        self.assertEqual(
            "fmla v0.2d,v1.2d,v2.2d\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64))

        self.assertEqual(
            "fmla v0.4s,v1.4s,v2.4s\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32))

        self.assertEqual(
            "fmla v0.8h,v1.8h,v2.8h\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16))

        self.assertEqual(
            "fmlalb v0.4s,v1.8h,v2.8h\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP32,
                    modifiers={mod.PART}, part=0))

        self.assertEqual(
            "fmlalt v0.4s,v1.8h,v2.8h\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP32,
                    modifiers={mod.PART}, part=1))

        self.assertEqual(
            "fmlsllbb v0.4s,v1.16b,v2.16b\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP8E5M2, b_dt=adt.FP8E5M2, c_dt=adt.FP32,
                    modifiers={mod.PART, mod.NP}, part=0))

        self.assertEqual(
            "fmlsllbt v0.4s,v1.16b,v2.16b\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP8E5M2, b_dt=adt.FP8E5M2, c_dt=adt.FP32,
                    modifiers={mod.PART, mod.NP}, part=1))

        self.assertEqual(
            "fmlslltb v0.4s,v1.16b,v2.16b\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP8E5M2, b_dt=adt.FP8E5M2, c_dt=adt.FP32,
                    modifiers={mod.PART, mod.NP}, part=2))

        self.assertEqual(
            "fmlslltt v0.4s,v1.16b,v2.16b\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP8E5M2, b_dt=adt.FP8E5M2, c_dt=adt.FP32,
                    modifiers={mod.PART, mod.NP}, part=3))
