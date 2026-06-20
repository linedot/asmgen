import unittest

from asmgen.asmblocks.operations import opdna1_action, opdna1_modifier as mod
from asmgen.registers import asm_data_type as adt
from asmgen.asmblocks.types.avx_types import (
    x86_greg,
    xmm_vreg, ymm_vreg, zmm_vreg,
    avx_freg,
    reg_prefixer
)
# Adjust imports based on your structure
from asmgen.asmblocks.avx_opdna1 import avx128_opdna1, avx256_opdna1, avx512_opdna1

def asmwrap(s: str) -> str:
    lines = s.split("\n")
    return "".join(f"{line}\n" for line in lines)

class test_avx_opdna1(unittest.TestCase):
    def setUp(self):
        self.r8 = x86_greg(0)
        self.r15 = x86_greg(7)
        
        self.xmm0 = xmm_vreg(0)
        self.ymm0 = ymm_vreg(self.xmm0.idx)
        self.zmm0 = zmm_vreg(self.xmm0.idx)
        self.xmm1 = xmm_vreg(1) 
        self.ymm1 = ymm_vreg(self.xmm1.idx) 
        self.zmm1 = zmm_vreg(self.xmm1.idx) 
        self.f0   = avx_freg(0) # scalar float

        self.output_inline = False
        self.asmwrap = asmwrap
        self.rpref = reg_prefixer(lambda : False)

        self.load_128  = avx128_opdna1(action=opdna1_action.LOAD, asmwrap=self.asmwrap, rpref=self.rpref)
        self.load_256  = avx256_opdna1(action=opdna1_action.LOAD, asmwrap=self.asmwrap, rpref=self.rpref)
        self.load_512  = avx512_opdna1(action=opdna1_action.LOAD, asmwrap=self.asmwrap, rpref=self.rpref)
        self.store_512 = avx512_opdna1(action=opdna1_action.STORE, asmwrap=self.asmwrap, rpref=self.rpref)

    # --- 1. Basic Routing & Unit Stride ---
    def test_scalar_routing(self):
        """ Ensure scalar registers fall back to x86_opdna1 """
        self.assertEqual(
            self.load_128(dregs=[self.f0], areg=self.r8, dt=adt.FP32, modifiers={}),
            f"vmovss (%r8), %xmm0\n"
        )

    def test_basic_vector_loads(self):
        """ Test standard vmovups / vmovupd """
        self.assertEqual(
            self.load_128(dregs=[self.xmm0], areg=self.r8, dt=adt.FP32, modifiers={}),
            f"vmovups (%r8), %xmm0\n"
        )
        self.assertEqual(
            self.load_512(dregs=[self.zmm1], areg=self.r15, dt=adt.FP64, modifiers={}),
            f"vmovupd (%r15), %zmm1\n"
        )

    def test_vector_offsets(self):
        """ VOFFSET scales by the SIMD byte size (16, 32, 64) """
        self.assertEqual(
            self.load_128(dregs=[self.xmm0], areg=self.r8, dt=adt.FP32, modifiers={mod.VOFFSET}, voffset=2),
            f"vmovups 32(%r8), %xmm0\n"
        )
        self.assertEqual(
            self.load_512(dregs=[self.zmm0], areg=self.r8, dt=adt.FP32, modifiers={mod.VOFFSET}, voffset=2),
            f"vmovups 128(%r8), %zmm0\n"
        )

    # --- 2. Broadcasts ---
    def test_broadcasts(self):
        self.assertEqual(
            self.load_256(dregs=[self.ymm0], areg=self.r8, dt=adt.FP32, modifiers={mod.BCAST}),
            f"vbroadcastss (%r8), %ymm0\n"
        )
        # BCAST should fail on STORE
        with self.assertRaisesRegex(ValueError, "BCAST modifier can only be used with loads"):
            self.store_512(dregs=[self.zmm0], areg=self.r8, dt=adt.FP32, modifiers={mod.BCAST})

    # --- 3. Lane Loads (Primitives) ---
    def test_lane_loads_fp64(self):
        """ Test FP64 vmovsd (lane 0) and vmovhpd (lane 1) """
        self.assertEqual(
            self.load_128(dregs=[self.xmm0], areg=self.r8, dt=adt.FP64, modifiers={mod.ILANE}, lane=0),
            f"vmovsd (%r8), %xmm0\n"
        )
        self.assertEqual(
            self.load_128(dregs=[self.xmm0], areg=self.r8, dt=adt.FP64, modifiers={mod.ILANE}, lane=1),
            f"vmovhpd (%r8), %xmm0, %xmm0\n"
        )

    def test_lane_loads_fp32(self):
        """ Test FP32 vmovss (lane 0) and vinsertps (lane > 0) """
        self.assertEqual(
            self.load_128(dregs=[self.xmm1], areg=self.r8, dt=adt.FP32, modifiers={mod.ILANE}, lane=0),
            f"vmovss (%r8), %xmm1\n"
        )
        # lane 1 << 4 = 0x10
        self.assertEqual(
            self.load_128(dregs=[self.xmm1], areg=self.r8, dt=adt.FP32, modifiers={mod.ILANE}, lane=1),
            f"vinsertps $16, (%r8), %xmm1, %xmm1\n"
        )

    # --- 4. Gather & Scatter (VINDEX) ---
    def test_gather_avx2(self):
        """ AVX2 Gather format """
        self.assertEqual(
            self.load_256(dregs=[self.ymm0], areg=self.r8, dt=adt.FP64, 
                          modifiers={mod.VINDEX}, vidxreg=self.xmm1, it=adt.SINT32),
            f"vgatherdpd (%r8), %xmm1, %ymm0\n"
        )

    def test_gather_avx512(self):
        """ AVX512 Gather with K-mask prefill """
        expected = "kxnorw %k2, %k2, %k2\nvgatherqps (%r8,%zmm1,1), %zmm0{%k2}\n"
        self.assertEqual(
            self.load_512(dregs=[self.zmm0], areg=self.r8, dt=adt.FP32, 
                          modifiers={mod.VINDEX}, vidxreg=self.zmm1, it=adt.SINT64),
            expected
        )

    def test_scatter_unsupported(self):
        """ AVX128/256 should reject scatter """
        with self.assertRaisesRegex(NotImplementedError, "no store with vector register stride"):
            store_128 = avx128_opdna1(action=opdna1_action.STORE,
                                      asmwrap=self.asmwrap,
                                      rpref=self.rpref)
            store_128(dregs=[self.xmm0], areg=self.r8, dt=adt.FP32, 
                      modifiers={mod.VINDEX}, vidxreg=self.xmm1, it=adt.SINT32)

if __name__ == '__main__':
    unittest.main()
