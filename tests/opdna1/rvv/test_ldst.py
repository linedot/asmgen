import unittest

from asmgen.asmblocks.operations import opdna1_modifier as mod
from asmgen.registers import asm_data_type as adt, asm_index_type as ait
from asmgen.asmblocks.types.riscv64_types import riscv64_greg, riscv64_freg
from asmgen.asmblocks.types.rvv_types import rvv_vreg
from asmgen.asmblocks.rvv import rvv  # Updated import

class test_rvv_opdna1(unittest.TestCase):
    def setUp(self):
        # We'll map these to standard indices (e.g., t0 is often t0)
        self.t0 = riscv64_greg(0) 
        self.t1 = riscv64_greg(1)
        self.f0 = riscv64_freg(0)
        
        # Populate a list of 32 vector registers to allow nice slicing like self.vs[:1]
        self.vs = [rvv_vreg(i) for i in range(32)]
        
        # Using the wrapper class directly
        self.rvv = rvv()

    def test_scalar_routing(self):
        """ Test that scalar registers are successfully routed through the rvv wrapper """
        self.assertEqual(
            self.rvv.load(dregs=[self.f0], areg=self.t0, dt=adt.FP32, modifiers={}),
            "flw f0, 0(t0)" # Assuming riscv64_greg(5) prints as t0
        )

    def test_unit_stride(self):
        """ Test contiguous unit-stride vector loads/stores """
        self.assertEqual(
            self.rvv.load(dregs=self.vs[:1], areg=self.t0, dt=adt.FP64, modifiers={}),
            "vle64.v v0, (t0)"
        )
        self.assertEqual(
            self.rvv.store(dregs=self.vs[:1], areg=self.t0, dt=adt.FP32, modifiers={}),
            "vse32.v v0, (t0)"
        )

    def test_strided_vector_operations(self):
        """ Test vector load/store with GSTRIDE (vlse/vsse) """
        self.assertEqual(
            self.rvv.load(dregs=self.vs[:1], areg=self.t0, dt=adt.FP64, 
                          modifiers={mod.GSTRIDE}, streg=self.t1),
            "vlse64.v v0, (t0), t1"
        )

    def test_indexed_vector_operations(self):
        """ Test unordered indexed vector operations (vluxei/vsuxei) """
        self.assertEqual(
            self.rvv.store(dregs=self.vs[:1], areg=self.t0,
                           dt=adt.FP32, it=ait.INT32,
                           modifiers={mod.VINDEX}, vidxreg=self.vs[2]),
            "vsuxei32.v v0, (t0), v2"
        )

    def test_structured_segment_operations(self):
        """ Test segment load operations (vlseg) """
        self.assertEqual(
            self.rvv.load(dregs=self.vs[:2], areg=self.t0, dt=adt.FP64, 
                          modifiers={mod.STRUCT}, nstructs=2),
            "vlseg2e64.v v0, (t0)"
        )

    def test_mutually_exclusive_modifiers(self):
        """ Test that GSTRIDE and VINDEX are rejected together """
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            self.rvv.load(dregs=self.vs[:1], areg=self.t0, dt=adt.FP32, 
                          modifiers={mod.GSTRIDE, mod.VINDEX}, streg=self.t1, vidxreg=self.vs[2])

    def test_lmul_contiguity(self):
        """ Test that passing non-contiguous segment registers raises an error """
        # Assuming default LMUL=1, passing v0 and v2 should fail
        with self.assertRaisesRegex(ValueError, "Segmented registers must be consecutive"):
            self.rvv.load(dregs=[self.vs[0], self.vs[2]], areg=self.t0, dt=adt.FP32, 
                          modifiers={mod.STRUCT}, nstructs=2)

    def test_missing_required_parameters(self):
        """ Test kwargs validation for VINDEX and STRUCT """
        with self.assertRaisesRegex(ValueError, "Missing parameter: vidxreg"):
            self.rvv.load(dregs=self.vs[:1], areg=self.t0, dt=adt.FP32, modifiers={mod.VINDEX})
            
        with self.assertRaisesRegex(ValueError, "Missing parameter: nstructs"):
            self.rvv.load(dregs=self.vs[:1], areg=self.t0, dt=adt.FP32, modifiers={mod.STRUCT})

if __name__ == '__main__':
    unittest.main()
