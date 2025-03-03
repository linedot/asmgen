import unittest

from asmgen.asmblocks.sme import sme
from asmgen.asmblocks.sve import sve
from asmgen.registers import asm_data_type as adt

from asmgen.asmblocks.operations import modifier as mod

class test_sve_opd3(unittest.TestCase):
    def test_fmul(self):
        gen = sve()
        gen.set_output_inline(yesno=False)

        self.assertEqual(
            "fmul z0.d,z1.d,z2.d\n",
            gen.fmul(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                     a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64))

        self.assertEqual(
            "fmul z0.s,z1.s,z2.s\n",
            gen.fmul(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                     a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32))

        self.assertEqual(
            "fmul z0.h,z1.h,z2.h\n",
            gen.fmul(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                     a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16))

        self.assertEqual(
            "smullb z0.s,z1.h,z2.h\n",
            gen.fmul(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                     a_dt=adt.SINT16, b_dt=adt.SINT16, c_dt=adt.SINT32,
                     modifiers={mod.part}, part=0))

        with self.assertRaises(ValueError):
            gen.fmul(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                     a_dt=adt.FP8E5M2, b_dt=adt.FP8E5M2, c_dt=adt.FP32,
                     modifiers={mod.part}, part=0)
