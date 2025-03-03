import unittest

from asmgen.asmblocks.neon import neon
from asmgen.registers import asm_data_type as adt

from asmgen.asmblocks.operations import modifier as mod

class test_neon_opd3(unittest.TestCase):
    def test_fmul(self):
        gen = neon()
        gen.set_output_inline(yesno=False)

        self.assertEqual(
            "fmul v0.2d,v1.2d,v2.2d\n",
            gen.fmul(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                     a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64))

        self.assertEqual(
            "fmul v0.4s,v1.4s,v2.4s\n",
            gen.fmul(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                     a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32))

        self.assertEqual(
            "fmul v0.8h,v1.8h,v2.8h\n",
            gen.fmul(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                     a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16))

        self.assertEqual(
            "mul v0.8h,v1.8h,v2.8h\n",
            gen.fmul(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                     a_dt=adt.SINT16, b_dt=adt.SINT16, c_dt=adt.SINT16))

        with self.assertRaises(ValueError):
            gen.fmul(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                     a_dt=adt.FP8E5M2, b_dt=adt.FP8E5M2, c_dt=adt.FP32,
                     modifiers={mod.part}, part=0)
