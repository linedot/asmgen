import unittest

from asmgen.asmblocks.sme import sme
from asmgen.registers import asm_data_type as adt

from asmgen.asmblocks.operations import modifier as mod

class test_sme_opd3(unittest.TestCase):
    def test_fopa(self):
        gen = sme()
        gen.set_output_inline(yesno=False)
        self.assertEqual(
            "fmopa za0.d,p0/m,p0/m,z0.d,z1.d\n",
            gen.fopa(adreg=gen.vreg(0),bdreg=gen.vreg(1),cdreg=gen.treg(0),
                a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64))
        self.assertEqual(
            "fmops za0.d,p0/m,p0/m,z0.d,z1.d\n",
            gen.fopa(adreg=gen.vreg(0),bdreg=gen.vreg(1),cdreg=gen.treg(0),
                a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64,
                     modifiers={mod.np}))
        self.assertEqual(
            "fmopa za0.s,p0/m,p0/m,z0.s,z1.s\n",
            gen.fopa(adreg=gen.vreg(0),bdreg=gen.vreg(1),cdreg=gen.treg(0),
                a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32))
        self.assertEqual(
            "fmops za0.s,p0/m,p0/m,z0.s,z1.s\n",
            gen.fopa(adreg=gen.vreg(0),bdreg=gen.vreg(1),cdreg=gen.treg(0),
                a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32,
                     modifiers={mod.np}))
        self.assertEqual(
            "fmopa za0.h,p0/m,p0/m,z0.h,z1.h\n",
            gen.fopa(adreg=gen.vreg(0),bdreg=gen.vreg(1),cdreg=gen.treg(0),
                a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16))
        self.assertEqual(
            "fmops za0.h,p0/m,p0/m,z0.h,z1.h\n",
            gen.fopa(adreg=gen.vreg(0),bdreg=gen.vreg(1),cdreg=gen.treg(0),
                a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16,
                     modifiers={mod.np}))
 
        with self.assertRaises(ValueError):
            gen.fopa(adreg=gen.vreg(0),bdreg=gen.vreg(1),cdreg=gen.treg(0),
                a_dt=adt.FP16, b_dt=adt.FP32, c_dt=adt.FP64)       

        with self.assertRaises(ValueError):
            gen.fopa(adreg=gen.vreg(0),bdreg=gen.vreg(1),cdreg=gen.treg(0),
                a_dt=adt.UINT64, b_dt=adt.UINT64, c_dt=adt.UINT64)

        with self.assertRaises(ValueError):
            gen.fopa(adreg=gen.vreg(0),bdreg=gen.vreg(1),cdreg=gen.treg(0),
                a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.UINT64)
