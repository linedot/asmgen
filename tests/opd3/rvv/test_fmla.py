import unittest

from asmgen.asmblocks.rvv import rvv
from asmgen.registers import asm_data_type as adt

from asmgen.asmblocks.operations import modifier as mod

class test_rvv_opd3(unittest.TestCase):
    def test_fmla(self):
        gen = rvv()
        gen.set_output_inline(yesno=False)

        self.assertEqual(
            "vfmacc.vv v0,v1,v2\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64))

        self.assertEqual(
            "vfmacc.vv v0,v1,v2\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32))

        self.assertEqual(
            "vfmacc.vv v0,v1,v2\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16))

        self.assertEqual(
            "vfwmacc.vv v0,v2,v3\n",
            gen.fma(adreg=gen.vreg(2),bdreg=gen.vreg(3),cdreg=gen.vreg(0),
                    a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP32))

        self.assertEqual(
            "vmacc.vv v0,v1,v2\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.SINT32, b_dt=adt.SINT32, c_dt=adt.SINT32))

        self.assertEqual(
            "vfwnmsac.vf v0,v2,f0\n",
            gen.fma(adreg=gen.vreg(2),bdreg=gen.freg(0),cdreg=gen.vreg(0),
                    a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP32,
                    modifiers={mod.np,mod.vf}))
