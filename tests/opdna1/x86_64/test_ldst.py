# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
import unittest

from asmgen.asmblocks.operations import opdna1_action, opdna1_modifier as mod
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.types.avx_types import x86_greg, avx_freg, reg_prefixer
from asmgen.asmblocks.x86_opdna1 import x86_load, x86_store

class test_x86_opdna1(unittest.TestCase):
    def setUp(self):
        self.r8 = x86_greg(0)
        self.r15 = x86_greg(7)
        
        self.xmm0_32 = avx_freg(0)
        self.xmm1_64 = avx_freg(1)
        self.xmm2_16 = avx_freg(2)


        output_inline = False
        
        self.load = x86_load(asmwrap = lambda s : f"{s}\n",
                             rpref=reg_prefixer(lambda: False))
        self.store = x86_store(asmwrap = lambda s : f"{s}\n",
                               rpref=reg_prefixer(lambda: False))

    def test_basic_integer_loads(self):
        """ Test scalar integer zero-offset loads across sizes """
        self.assertEqual(
            self.load(dregs=[self.r8], areg=self.r15, dt=adt.UINT8, modifiers={}),
            f"movb (%r15), %r8\n"
        )
        self.assertEqual(
            self.load(dregs=[self.r8], areg=self.r15, dt=adt.UINT16, modifiers={}),
            f"movw (%r15), %r8\n"
        )
        self.assertEqual(
            self.load(dregs=[self.r8], areg=self.r15, dt=adt.UINT32, modifiers={}),
            f"movl (%r15), %r8\n"
        )
        self.assertEqual(
            self.load(dregs=[self.r8], areg=self.r15, dt=adt.UINT64, modifiers={}),
            f"movq (%r15), %r8\n"
        )

    def test_basic_float_loads(self):
        """ Test scalar floating point zero-offset loads """
        self.assertEqual(
            self.load(dregs=[self.xmm2_16], areg=self.r15, dt=adt.FP16, modifiers={}),
            f"vmovsh (%r15), %xmm2\n"
        )
        self.assertEqual(
            self.load(dregs=[self.xmm0_32], areg=self.r15, dt=adt.FP32, modifiers={}),
            f"vmovss (%r15), %xmm0\n"
        )
        self.assertEqual(
            self.load(dregs=[self.xmm1_64], areg=self.r15, dt=adt.FP64, modifiers={}),
            f"vmovsd (%r15), %xmm1\n"
        )

    def test_immediate_offset_stores(self):
        """ Test scalar stores with immediate offsets """
        self.assertEqual(
            self.store(dregs=[self.r8], areg=self.r15, dt=adt.UINT64, 
                       modifiers={mod.IOFFSET}, ioffset=32),
            f"movq %r8, 32(%r15)\n"
        )
        self.assertEqual(
            self.store(dregs=[self.xmm1_64], areg=self.r15, dt=adt.FP64, 
                       modifiers={mod.IOFFSET}, ioffset=-16),
            f"vmovsd %xmm1, -16(%r15)\n"
        )

    def test_invalid_modifiers(self):
        """ Test that invalid modifiers are properly rejected """
        invalid_mods = [mod.TINDEX, mod.VINDEX, mod.GLANE, mod.ILANE, mod.POSTINC, 
                        mod.TOFFSET, mod.VOFFSET, mod.ISTRIDE, mod.GSTRIDE, 
                        mod.STRUCT, mod.BCAST, mod.MASK]
        
        for invalid_mod in invalid_mods:
            with self.subTest(modifier=invalid_mod):
                with self.assertRaisesRegex(ValueError, "Base X86 has no"):
                    self.load(dregs=[self.r8], areg=self.r15, dt=adt.UINT64, modifiers={invalid_mod})

    def test_missing_ioffset(self):
        """ Test kwargs validation for IOFFSET """
        with self.assertRaisesRegex(ValueError, "Missing one of these parameters: ioffset"):
            self.load(dregs=[self.r8], areg=self.r15, dt=adt.UINT64, modifiers={mod.IOFFSET})

    def test_invalid_register_count(self):
        """ Verify exactly one register is strictly enforced """
        with self.assertRaisesRegex(ValueError,
                                    "Invalid data type combination: adreg:UINT64, bdreg:UINT64"):
            self.load(dregs=[self.r8, self.r15], areg=self.r15, dt=adt.UINT64, modifiers={})

if __name__ == '__main__':
    unittest.main()
