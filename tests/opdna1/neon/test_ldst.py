import unittest

from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.operations import opdna1_action, opdna1_modifier as mod
from asmgen.asmblocks.types.aarch64_types import aarch64_greg, aarch64_freg
from asmgen.asmblocks.types.neon_types import neon_vreg
from asmgen.registers import asm_data_type as adt

class test_neon_opdna1(unittest.TestCase):

    def setUp(self):
        # General purpose address registers
        self.x0 = aarch64_greg(0)
        self.x1 = aarch64_greg(1)

        # NEON vector registers
        self.v0 = neon_vreg(0)
        self.v1 = neon_vreg(1)
        self.v2 = neon_vreg(2)
        
        # Scalar register for routing test
        self.f0 = aarch64_freg(0, adt.FP32)

        # Instantiating the operations
        self.gen = neon()
        self.gen.set_output_inline(False)

    # --- 1. Scalar Routing ---

    def test_scalar_routing(self):
        """ Test that passing a scalar register routes correctly to the parent class """
        self.assertEqual(
            self.gen.load(dregs=[self.f0], areg=self.x0, dt=adt.FP32, modifiers={}),
            "ldr s0, [x0]\n"
        )

    # --- 2. Q-Register LDR/STR (VOFFSET / IOFFSET) ---

    def test_q_register_ioffset(self):
        """ Test that IOFFSET triggers the ldr/str qX mnemonic """
        self.assertEqual(
            self.gen.load(dregs=[self.v0], areg=self.x0, dt=adt.FP32, 
                      modifiers={mod.IOFFSET}, ioffset=16),
            "ldr q0, [x0, #16]\n"
        )

    def test_q_register_voffset(self):
        """ Test that VOFFSET calculates byte offset (voffset * 16) """
        self.assertEqual(
            self.gen.store(dregs=[self.v0], areg=self.x0, dt=adt.FP32, 
                       modifiers={mod.VOFFSET}, voffset=2),
            "str q0, [x0, #32]\n"
        )

    # --- 3. NEON Structural Loads (LD1, LD2, etc.) ---

    def test_basic_ld1(self):
        """ Test standard single-vector load (ld1) """
        self.assertEqual(
            self.gen.load(dregs=[self.v0], areg=self.x0, dt=adt.FP32, modifiers={}),
            "ld1 {v0.4s}, [x0]\n"
        )

    def test_struct_ld2(self):
        """ Test structured load into 2 registers """
        self.assertEqual(
            self.gen.load(dregs=[self.v0, self.v1], areg=self.x0, dt=adt.FP64, 
                      modifiers={mod.STRUCT}, nstructs=2),
            "ld2 {v0.2d, v1.2d}, [x0]\n"
        )

    # --- 4. Lane and Broadcast Loads ---

    def test_lane_load(self):
        """ Test loading into a specific lane (ILANE) """
        self.assertEqual(
            self.gen.load(dregs=[self.v0], areg=self.x0, dt=adt.FP32, 
                      modifiers={mod.ILANE}, lane=1),
            "ld1 {v0.s}[1], [x0]\n"
        )

    def test_broadcast_load(self):
        """ Test broadcasting a single element to all lanes (BCAST) """
        self.assertEqual(
            self.gen.load(dregs=[self.v0], areg=self.x0, dt=adt.FP16, 
                      modifiers={mod.BCAST}),
            "ld1r {v0.8h}, [x0]\n"
        )

    # --- 5. Addressing Mods ---

    def test_post_increment(self):
        """ Test post-increment addressing with structures """
        self.assertEqual(
            self.gen.load(dregs=[self.v0, self.v1], areg=self.x0, dt=adt.FP32, 
                      modifiers={mod.STRUCT, mod.POSTINC}, nstructs=2, iinc=32),
            "ld2 {v0.4s, v1.4s}, [x0], #32\n"
        )

    # --- 6. Error Handling & Validation ---

    def test_non_contiguous_registers(self):
        """ Segmented structural loads must use contiguous registers """
        with self.assertRaisesRegex(ValueError, "segmented registers must be contiguous"):
            self.gen.load(dregs=[self.v0, self.v2], areg=self.x0, dt=adt.FP32, 
                      modifiers={mod.STRUCT}, nstructs=2)

    def test_mutually_exclusive_modifiers(self):
        """ Test validation of incompatible modifiers """
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            self.gen.load(dregs=[self.v0], areg=self.x0, dt=adt.FP32, 
                      modifiers={mod.IOFFSET, mod.BCAST}, ioffset=16)

    def test_bcast_on_store(self):
        """ BCAST should only be allowed on LOAD operations """
        with self.assertRaisesRegex(ValueError, "only valid for LOAD"):
            self.gen.store(dregs=[self.v0], areg=self.x0, dt=adt.FP32, modifiers={mod.BCAST})

    def test_missing_required_params_sorted(self):
        """ Verify Option 1 sorting is working for missing parameters """
        with self.assertRaisesRegex(ValueError, "Missing one of these parameters: iinc, increg"):
            self.gen.load(dregs=[self.v0], areg=self.x0, dt=adt.FP32, modifiers={mod.POSTINC})

if __name__ == '__main__':
    unittest.main()
