import unittest

from asmgen.asmblocks.operations import opdna1_action, opdna1_modifier as mod
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.types.aarch64_types import aarch64_greg, aarch64_freg
from asmgen.asmblocks.types.sve_types import sve_vreg, sve_preg
from asmgen.asmblocks.sve_opdna1 import sve_load, sve_store

def asmwrap(s: str) -> str:
    return f"{s}\n"

class test_sve_opdna1(unittest.TestCase):
    def setUp(self):
        self.x0 = aarch64_greg(0)
        self.x1 = aarch64_greg(1)
        
        self.z0 = sve_vreg(0)
        self.z1 = sve_vreg(1)
        self.p1 = sve_preg(1)
        
        self.f0 = aarch64_freg(0,dt=adt.FP32)

        self.load = sve_load(asmwrap=asmwrap)
        self.store = sve_store(asmwrap=asmwrap)

    def test_scalar_routing(self):
        """ Ensure scalar registers route back to aarch64_opdna1 """
        self.assertEqual(
            self.load(dregs=[self.x1], areg=self.x0, dt=adt.UINT64, modifiers={}),
            "ldr x1, [x0]\n"
        )
        self.assertEqual(
            self.load(dregs=[self.f0], areg=self.x0, dt=adt.FP32, modifiers={}),
            "ldr s0, [x0]\n"
        )

    def test_basic_vector_ops(self):
        """ Test standard unit-stride loads and stores (ld1w, st1d) """
        # FP32 -> memory suffix 'w', element suffix '.s'
        self.assertEqual(
            self.load(dregs=[self.z0], areg=self.x0, dt=adt.FP32, modifiers={}),
            "ld1w {z0.s}, p0/z, [x0]\n"
        )
        # FP64 -> memory suffix 'd', element suffix '.d'. Store uses un-zeroed predicate.
        self.assertEqual(
            self.store(dregs=[self.z1], areg=self.x0, dt=adt.FP64, modifiers={}),
            "st1d {z1.d}, p0, [x0]\n"
        )

    def test_explicit_predicate(self):
        """ Test passing an explicit predicate register via kwargs """
        self.assertEqual(
            self.load(dregs=[self.z0], areg=self.x0, dt=adt.FP32, modifiers={}, preg=self.p1),
            "ld1w {z0.s}, p1/z, [x0]\n"
        )

    def test_voffset_mul_vl(self):
        """ Test that VOFFSET calculates MUL VL """
        self.assertEqual(
            self.load(dregs=[self.z0], areg=self.x0, dt=adt.FP32, modifiers={mod.VOFFSET}, voffset=3),
            "ld1w {z0.s}, p0/z, [x0, #3, MUL VL]\n"
        )

    def test_goffset_scaling(self):
        """ Test scalar offset with LSL scaling """
        # FP64 (8 bytes) -> log2(8) = 3 -> lsl #3
        self.assertEqual(
            self.load(dregs=[self.z0], areg=self.x0, dt=adt.FP64, modifiers={mod.GOFFSET}, offreg=self.x1),
            "ld1d {z0.d}, p0/z, [x0, x1, lsl #3]\n"
        )

    def test_vindex_gather(self):
        """ Test VINDEX gather with element scaling """
        # FP32 data, SINT64 indices -> requires z1.d index scaling by lsl #2
        self.assertEqual(
            self.load(dregs=[self.z0], areg=self.x0, dt=adt.FP32, 
                      modifiers={mod.VINDEX}, vidxreg=self.z1, it=adt.SINT64),
            "ld1w {z0.s}, p0/z, [x0, z1.d, lsl #2]\n"
        )

    def test_broadcast(self):
        """ Test BCAST generates ld1r """
        self.assertEqual(
            self.load(dregs=[self.z0], areg=self.x0, dt=adt.FP32, modifiers={mod.BCAST}),
            "ld1rw {z0.s}, p0/z, [x0]\n"
        )

    def test_structures(self):
        """ Test structured loads (ld2, ld3) """
        self.assertEqual(
            self.load(dregs=[self.z0, self.z1], areg=self.x0, dt=adt.FP32, 
                      modifiers={mod.STRUCT}, nstructs=2),
            "ld2w {z0.s, z1.s}, p0/z, [x0]\n"
        )

if __name__ == '__main__':
    unittest.main()
