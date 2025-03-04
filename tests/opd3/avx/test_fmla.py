import unittest

from asmgen.asmblocks.avx_fma import fma128,fma256,avx512
from asmgen.registers import asm_data_type as adt

from asmgen.asmblocks.operations import modifier as mod

class test_avx_opd3(unittest.TestCase):
    def test_fmla(self):
        gen128 = fma128()
        gen256 = fma256()
        gen512 = avx512()
        gen128.set_output_inline(yesno=False)
        gen256.set_output_inline(yesno=False)
        gen512.set_output_inline(yesno=False)

        self.assertEqual(
            "vfmadd231pd %%xmm1,%%xmm2,%%xmm0\n",
            gen128.fma(adreg=gen128.vreg(1),bdreg=gen128.vreg(2),cdreg=gen128.vreg(0),
                       a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64))
        self.assertEqual(
            "vfmadd231pd %%ymm1,%%ymm2,%%ymm0\n",
            gen256.fma(adreg=gen256.vreg(1),bdreg=gen256.vreg(2),cdreg=gen256.vreg(0),
                       a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64))
        self.assertEqual(
            "vfmadd231pd %%zmm1,%%zmm2,%%zmm0\n",
            gen512.fma(adreg=gen512.vreg(1),bdreg=gen512.vreg(2),cdreg=gen512.vreg(0),
                       a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64))

        self.assertEqual(
            "vfmadd231ps %%xmm1,%%xmm2,%%xmm0\n",
            gen128.fma(adreg=gen128.vreg(1),bdreg=gen128.vreg(2),cdreg=gen128.vreg(0),
                       a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32))
        self.assertEqual(
            "vfmadd231ps %%ymm1,%%ymm2,%%ymm0\n",
            gen256.fma(adreg=gen256.vreg(1),bdreg=gen256.vreg(2),cdreg=gen256.vreg(0),
                       a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32))
        self.assertEqual(
            "vfmadd231ps %%zmm1,%%zmm2,%%zmm0\n",
            gen512.fma(adreg=gen512.vreg(1),bdreg=gen512.vreg(2),cdreg=gen512.vreg(0),
                       a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32))

        self.assertEqual(
            "vfmadd231ph %%zmm1,%%zmm2,%%zmm0\n",
            gen512.fma(adreg=gen512.vreg(1),bdreg=gen512.vreg(2),cdreg=gen512.vreg(0),
                       a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16))

