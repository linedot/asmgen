# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Tests SVE fmla instruction code generation
"""
import unittest

from asmgen.asmblocks.sve import sve
from asmgen.registers import asm_data_type as adt

from asmgen.asmblocks.operations import modifier as mod

class test_sve_opd3(unittest.TestCase):
    """
    Tests SVE opd3 operations
    """
    def test_fmla(self):
        """
        Tests that the SVE generator generates correct FMA instructions
        """
        gen = sve()
        gen.set_output_inline(yesno=False)

        self.assertEqual(
            "fmla z0.d,p0/m,z1.d,z2.d\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64))

        self.assertEqual(
            "fmla z0.s,p0/m,z1.s,z2.s\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32))

        self.assertEqual(
            "fmla z0.h,p0/m,z1.h,z2.h\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16))

        self.assertEqual(
            "fmlalb z0.s,p0/m,z1.h,z2.h\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP32,
                    modifiers={mod.PART}, part=0))

        self.assertEqual(
            "fmlalt z0.s,p0/m,z1.h,z2.h\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP32,
                    modifiers={mod.PART}, part=1))

        self.assertEqual(
            "fmlsllbb z0.s,p0/m,z1.b,z2.b\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP8E5M2, b_dt=adt.FP8E5M2, c_dt=adt.FP32,
                    modifiers={mod.PART, mod.NP}, part=0))

        self.assertEqual(
            "fmlsllbt z0.s,p0/m,z1.b,z2.b\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP8E5M2, b_dt=adt.FP8E5M2, c_dt=adt.FP32,
                    modifiers={mod.PART, mod.NP}, part=1))

        self.assertEqual(
            "fmlslltb z0.s,p0/m,z1.b,z2.b\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP8E5M2, b_dt=adt.FP8E5M2, c_dt=adt.FP32,
                    modifiers={mod.PART, mod.NP}, part=2))

        self.assertEqual(
            "fmlslltt z0.s,p0/m,z1.b,z2.b\n",
            gen.fma(adreg=gen.vreg(1),bdreg=gen.vreg(2),cdreg=gen.vreg(0),
                    a_dt=adt.FP8E5M2, b_dt=adt.FP8E5M2, c_dt=adt.FP32,
                    modifiers={mod.PART, mod.NP}, part=3))
