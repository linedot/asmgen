# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Tests SVE fmul instruction code generation
"""
import unittest

from asmgen.asmblocks.sve import sve
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.operations import opd3_modifier as mod

from .test_sve_opd3 import test_sve_opd3

class test_sve_opd3(test_sve_opd3):
    """
    Tests SVE opd3 operations
    """
    def test_fmul(self):
        """
        Tests that the SVE generator generates correct FMUL instructions
        """

        self.assertEqual(
                "fmul z0.d,p0/m,z1.d,z2.d\n",
                self.gen.fmul(adreg=self.gen.vreg(1),
                              bdreg=self.gen.vreg(2),
                              cdreg=self.gen.vreg(0),
                              a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64))

        self.assertEqual(
                "fmul z0.s,p0/m,z1.s,z2.s\n",
                self.gen.fmul(adreg=self.gen.vreg(1),
                              bdreg=self.gen.vreg(2),
                              cdreg=self.gen.vreg(0),
                              a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32))

        self.assertEqual(
                "fmul z0.h,p0/m,z1.h,z2.h\n",
                self.gen.fmul(adreg=self.gen.vreg(1),
                              bdreg=self.gen.vreg(2),
                              cdreg=self.gen.vreg(0),
                              a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16))

        self.assertEqual(
                "smullb z0.s,z1.h,z2.h\n",
                self.gen.fmul(adreg=self.gen.vreg(1),
                              bdreg=self.gen.vreg(2),
                              cdreg=self.gen.vreg(0),
                              a_dt=adt.SINT16, b_dt=adt.SINT16, c_dt=adt.SINT32,
                              modifiers={mod.PART}, part=0))

        with self.assertRaises(ValueError):
            self.gen.fmul(adreg=self.gen.vreg(1),
                          bdreg=self.gen.vreg(2),
                          cdreg=self.gen.vreg(0),
                          a_dt=adt.FP8E5M2, b_dt=adt.FP8E5M2, c_dt=adt.FP32,
                          modifiers={mod.PART}, part=0)

    def test_wrong_registers(self):
        """
        Tests that the correct error is raised if wrong registers are passed
        to the operation
        """
        with self.assertRaisesRegex(
                ValueError,
                "All dregs of a SVE opd3 must be sve_vreg"):
            self.gen.fmul(
                adreg=self.gen.vreg(1),
                bdreg=self.gen.vreg(2),
                cdreg=self.gen.freg(0,dt=adt.FP64),
                a_dt=adt.FP64,b_dt=adt.FP64,c_dt=adt.FP64)
