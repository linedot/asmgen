# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
import unittest

from asmgen.asmblocks.operations import opdna1_action, opdna1_modifier as mod
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.types.avx_types import x86_greg, avx_freg
from asmgen.asmblocks.x86_opdna1 import x86_load, x86_store

class test_x86_opdna1(unittest.TestCase):
    def setUp(self):
        self.rax = x86_greg(0) # e.g., %rax
        self.rdi = x86_greg(7) # e.g., %rdi
        
        self.xmm0_32 = avx_freg(0)
        self.xmm1_64 = avx_freg(1)
        
        self.load = x86_load()
        self.store = x86_store()

    def test_basic_integer_loads(self):
        """ Test scalar integer zero-offset loads across sizes """
        self.assertEqual(
            self.load(dregs=[self.rax], areg=self.rdi, dt=adt.UINT8, modifiers={}),
            f"movb ({self.rdi}), {self.rax}"
        )
        self.assertEqual(
            self.load(dregs=[self.rax], areg=self.rdi, dt=adt.UINT16, modifiers={}),
            f"movw ({self.rdi}), {self.rax}"
        )
        self.assertEqual(
            self.load(dregs=[self.rax], areg=self.rdi, dt=adt.UINT32, modifiers={}),
            f"movl ({self.rdi}), {self.rax}"
        )
        self.assertEqual(
            self.load(dregs=[self.rax], areg=self.rdi, dt=adt.UINT64, modifiers={}),
            f"movq ({self.rdi}), {self.rax}"
        )

    def test_basic_float_loads(self):
        """ Test scalar floating point zero-offset loads """
        self.assertEqual(
            self.load(dregs=[self.xmm0_32], areg=self.rdi, dt=adt.FP32, modifiers={}),
            f"vmovss ({self.rdi}), {self.xmm0_32}"
        )
        self.assertEqual(
            self.load(dregs=[self.xmm1_64], areg=self.rdi, dt=adt.FP64, modifiers={}),
            f"vmovsd ({self.rdi}), {self.xmm1_64}"
        )

    def test_immediate_offset_stores(self):
        """ Test scalar stores with immediate offsets """
        self.assertEqual(
            self.store(dregs=[self.rax], areg=self.rdi, dt=adt.UINT64, 
                       modifiers={mod.IOFFSET}, ioffset=32),
            f"movq {self.rax}, 32({self.rdi})"
        )
        self.assertEqual(
            self.store(dregs=[self.xmm1_64], areg=self.rdi, dt=adt.FP64, 
                       modifiers={mod.IOFFSET}, ioffset=-16),
            f"vmovsd {self.xmm1_64}, -16({self.rdi})"
        )

    def test_invalid_modifiers(self):
        """ Test that invalid modifiers are properly rejected """
        invalid_mods = [mod.TINDEX, mod.VINDEX, mod.GLANE, mod.ILANE, mod.POSTINC, 
                        mod.TOFFSET, mod.VOFFSET, mod.ISTRIDE, mod.GSTRIDE, 
                        mod.STRUCT, mod.BCAST, mod.MASK]
        
        for invalid_mod in invalid_mods:
            with self.subTest(modifier=invalid_mod):
                with self.assertRaisesRegex(ValueError, "Base X86 has no"):
                    self.load(dregs=[self.rax], areg=self.rdi, dt=adt.UINT64, modifiers={invalid_mod})

    def test_missing_ioffset(self):
        """ Test kwargs validation for IOFFSET """
        with self.assertRaisesRegex(ValueError, "Missing parameter: ioffset"):
            self.load(dregs=[self.rax], areg=self.rdi, dt=adt.UINT64, modifiers={mod.IOFFSET})

    def test_invalid_register_count(self):
        """ Verify exactly one register is strictly enforced """
        with self.assertRaisesRegex(ValueError, "exactly one register"):
            self.load(dregs=[self.rax, self.rdi], areg=self.rdi, dt=adt.UINT64, modifiers={})

if __name__ == '__main__':
    unittest.main()
