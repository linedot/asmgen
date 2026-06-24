# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Tests SME fopa instruction code generation
"""
import unittest

from asmgen.asmblocks.sme import sme
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.operations import opd3_modifier as mod

class test_sme_opd3(unittest.TestCase):
    """
    Tests SME opd3 operations broken down by category.
    """

    def setUp(self):
        """
        Initialize the generator before each test to reduce boilerplate.
        """
        self.gen = sme()
        self.gen.set_output_inline(yesno=False)
        self.za0 = self.gen.treg(0, adt.FP64) # Default, will override dt in tests
        self.z0 = self.gen.vreg(0)
        self.z1 = self.gen.vreg(1)

    # ---------------------------------------------------------
    # Float Valid Operations
    # ---------------------------------------------------------
    
    def test_fopa_float_same_size(self):
        """Tests standard same-size float outer product accumulate (fmopa)"""
        # FP64
        self.assertEqual(
            "fmopa za0.d,p0/m,p0/m,z0.d,z1.d\n",
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.FP64),
                          a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64)
        )
        # FP32
        self.assertEqual(
            "fmopa za0.s,p0/m,p0/m,z0.s,z1.s\n",
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.FP32),
                          a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP32)
        )
        # FP16
        self.assertEqual(
            "fmopa za0.h,p0/m,p0/m,z0.h,z1.h\n",
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.FP16),
                          a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP16)
        )

    def test_fopa_float_widening(self):
        """Tests widening float outer product accumulate"""
        # FP16 -> FP32
        self.assertEqual(
            "fmopa za0.s,p0/m,p0/m,z0.h,z1.h\n",
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.FP32),
                          a_dt=adt.FP16, b_dt=adt.FP16, c_dt=adt.FP32)
        )

    # ---------------------------------------------------------
    # Integer Valid Operations
    # ---------------------------------------------------------

    def test_fopa_int_signed(self):
        """Tests signed integer outer product accumulate (smopa)"""
        # SINT16 -> SINT32
        self.assertEqual(
            "smopa za0.s,p0/m,p0/m,z0.h,z1.h\n",
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.SINT32),
                          a_dt=adt.SINT16, b_dt=adt.SINT16, c_dt=adt.SINT32)
        )
        # SINT8 -> SINT32
        self.assertEqual(
            "smopa za0.s,p0/m,p0/m,z0.b,z1.b\n",
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.SINT32),
                          a_dt=adt.SINT8, b_dt=adt.SINT8, c_dt=adt.SINT32)
        )

    def test_fopa_int_unsigned(self):
        """Tests unsigned integer outer product accumulate (umopa)"""
        # UINT16 -> UINT32
        self.assertEqual(
            "umopa za0.s,p0/m,p0/m,z0.h,z1.h\n",
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.UINT32),
                          a_dt=adt.UINT16, b_dt=adt.UINT16, c_dt=adt.UINT32)
        )

    def test_fopa_mixed_sign_int_bug(self):
        """
        Tests the mixed sign INT constraints.
        """
        # SINT16,UINT16 -> SINT32
        self.assertEqual(
            "sumopa za0.s,p0/m,p0/m,z0.h,z1.h\n",
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.SINT32),
                          a_dt=adt.SINT16, b_dt=adt.UINT16, c_dt=adt.SINT32)
        )

    # ---------------------------------------------------------
    # Modifiers
    # ---------------------------------------------------------

    def test_fopa_subtract_modifier(self):
        """Tests that the Negative Product (NP) modifier generates subtract instructions (mops)"""
        # Float
        self.assertEqual(
            "fmops za0.d,p0/m,p0/m,z0.d,z1.d\n",
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.FP64),
                          a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64,
                          modifiers={mod.NP})
        )
        # Signed Int
        self.assertEqual(
            "smops za0.s,p0/m,p0/m,z0.h,z1.h\n",
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.SINT32),
                          a_dt=adt.SINT16, b_dt=adt.SINT16, c_dt=adt.SINT32,
                          modifiers={mod.NP})
        )

    # ---------------------------------------------------------
    # Exceptions and Error Handling
    # ---------------------------------------------------------

    def test_fopa_invalid_modifiers(self):
        """Tests that unsupported SME modifiers raise an error"""
        invalid_mods = [mod.IDX, mod.PART, mod.VF]
        for m in invalid_mods:
            with self.assertRaisesRegex(ValueError, "unsupported modifiers for SME"):
                self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.za0,
                              a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64,
                              modifiers={m})

    def test_fopa_invalid_registers(self):
        """Tests that passing the wrong register types (e.g. vreg for ZA) raises errors"""
        # C register is not an sme_treg
        with self.assertRaisesRegex(ValueError, "is not an sme_treg"):
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.vreg(0),
                          a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64)
        
        # A register is not an sve_vreg
        with self.assertRaisesRegex(ValueError, "is not an sve_vreg"):
            self.gen.fopa(adreg=self.za0, bdreg=self.z1, cdreg=self.za0,
                          a_dt=adt.FP64, b_dt=adt.FP64, c_dt=adt.FP64)

    def test_fopa_invalid_types_and_sizes(self):
        """Tests logical restrictions on datatypes inside __call__"""
        
        # C is smaller than A/B
        with self.assertRaisesRegex(ValueError,
                "Unsupported type combination a=asm_data_type.SINGLE,b=asm_data_type.SINGLE,c=asm_data_type.HALF"):
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.FP16),
                          a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.FP16)

        # Mixing Float and Int
        with self.assertRaisesRegex(ValueError,
                "Unsupported type combination a=asm_data_type.SINGLE,b=asm_data_type.SINGLE,c=asm_data_type.UINT64"):
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.UINT64),
                          a_dt=adt.FP32, b_dt=adt.FP32, c_dt=adt.UINT64)

        # C is not in valid_c_types (e.g. UINT8)
        with self.assertRaisesRegex(ValueError,
                "Unsupported type combination a=asm_data_type.UINT8,b=asm_data_type.UINT8,c=asm_data_type.UINT8"):
            self.gen.fopa(adreg=self.z0, bdreg=self.z1, cdreg=self.gen.treg(0, adt.UINT8),
                          a_dt=adt.UINT8, b_dt=adt.UINT8, c_dt=adt.UINT8)
