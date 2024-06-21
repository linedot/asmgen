from typing import Union
from asmgen.asmblocks.noarch import asmgen, freg, greg, vreg
from asmgen.asmblocks.noarch import asm_data_type as dt
from asmgen.asmblocks.noarch import asm_index_type as it

from asmgen.asmblocks.avx_fma import fma128,fma256,avx512
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.rvv071 import rvv071

import unittest
import re
from parameterized import parameterized, parameterized_class


allowed_not_implemented = {
        'rvv' : ['load_vector_dist1_inc',
                 'load_vector_dist1_boff',
                 'load_vector_voff',
                 'load_vector_immstride',
                 'store_vector_voff',
                 'store_vector_immstride',
                 'add_greg_voff',
                 'fma_idx'],
        'rvv071' : ['load_vector_dist1_inc',
                 'load_vector_dist1_boff',
                 'load_vector_voff',
                 'load_vector_immstride',
                 'store_vector_voff',
                 'store_vector_immstride',
                 'add_greg_voff',
                 'fma_idx'],
        'sve' : ['load_vector_dist1_inc',
                  'load_vector_immstride',
                  'load_vector_gregstride',
                  'store_vector_immstride',
                  'store_vector_gregstride',
                 'fma_vf'],
        'neon' : ['load_vector_dist1_boff',
                  'load_vector_immstride',
                  'load_vector_gregstride',
                  'load_vector_gather',
                  'store_vector_immstride',
                  'store_vector_gregstride',
                  'store_vector_scatter',
                  'fma_vf'],
        'fma128' : ['load_vector_dist1_inc',
                    'load_vector_immstride',
                    'load_vector_gregstride',
                    'store_vector_immstride',
                    'store_vector_gregstride',
                    'store_vector_scatter',
                    'fma_idx',
                    'fma_vf'],
        'fma256' : ['load_vector_dist1_inc',
                    'load_vector_immstride',
                    'load_vector_gregstride',
                    'store_vector_immstride',
                    'store_vector_gregstride',
                    'store_vector_scatter',
                    'fma_idx',
                    'fma_vf'],
        'avx512' : ['load_vector_dist1_inc',
                    'load_vector_immstride',
                    'load_vector_gregstride',
                    'store_vector_immstride',
                    'store_vector_gregstride',
                    'fma_idx',
                    'fma_vf']
}

@parameterized_class([
    {"name": "fma128", "gen": fma128(), "allowed": allowed_not_implemented['fma128']},
    {"name": "fma256", "gen": fma256(), "allowed": allowed_not_implemented['fma256']},
    {"name": "avx512", "gen": avx512(), "allowed": allowed_not_implemented['avx512']},
    {"name": "neon", "gen": neon(), "allowed": allowed_not_implemented['neon']},
    {"name": "sve", "gen": sve(), "allowed": allowed_not_implemented['sve']},
    {"name": "rvv", "gen": rvv(), "allowed": allowed_not_implemented['rvv']},
    {"name": "rvv071", "gen": rvv071(), "allowed": allowed_not_implemented['rvv071']},
])
class asm_implementation_test(unittest.TestCase):
    @parameterized.expand([
        ['simd_size'],
        ['c_simd_size_function'],
        ['is_vla'],
        ['indexable_elements', dt.DOUBLE],
        ['max_vregs'],
        ['max_gregs'],
        ['max_fregs'],
        ['min_prefetch_offset'],
        ['max_prefetch_offset'],
        ['min_load_voff'],
        ['max_load_voff'],
        ['min_load_immoff', dt.DOUBLE],
        ['max_load_immoff', dt.DOUBLE],
        ['min_load_immoff', dt.SINGLE],
        ['max_load_immoff', dt.SINGLE],
        ['freg', 0],
        ['greg', 0],
        ['vreg', 0],
        # We start doing a little trick, because we can't use self here:
        # g0 -> self.gen.greg(0), v0 -> self.gen.vreg(0), f0 -> self.gen.freg(0)
        ['loopbegin', 'g0', 'someloop'],
        ['loopend', 'g0', 'someloop'],
        ['fma', 'v0', 'v1', 'v2', dt.DOUBLE],
        ['fma', 'v0', 'v1', 'v2', dt.SINGLE],
        ['fma_vf', 'v0', 'f1', 'v1', dt.DOUBLE],
        ['fma_vf', 'v0', 'f1', 'v1', dt.SINGLE],
        ['fma_idx', 'v0', 'v1', 'v2', 0, dt.DOUBLE],
        ['fma_idx', 'v0', 'v1', 'v2', 0, dt.SINGLE],
        ['zero_greg', 'g0'],
        ['mov_greg', 'g0', 'g1'],
        ['mov_greg_to_param', 'g0', 'someparam'],
        ['mov_param_to_greg', 'someparam', 'g0'],
        ['mov_param_to_greg_shift', 'someparam', 'g0', 2],
        ['mov_greg_imm', 'g0', 40],
        ['add_greg_imm', 'g0', 40],
        ['add_greg_voff', 'g0', 40, dt.DOUBLE],
        ['add_greg_voff', 'g0', 40, dt.SINGLE],
        ['shift_greg_left', 'g0', 4],
        ['shift_greg_right', 'g0', 4],
        ['zero_vreg', 'v0', dt.DOUBLE],
        ['zero_vreg', 'v0', dt.SINGLE],
        ['prefetch_l1_boff', 'g0', 256],
        ['load_pointer', 'g0', 'someparam'],
        ['load_vector_voff', 'g0', 4, 'v0', dt.DOUBLE],
        ['load_vector_voff', 'g0', 4, 'v0', dt.SINGLE],
        ['load_scalar_immoff', 'g0', 4, 'f0', dt.DOUBLE],
        ['load_scalar_immoff', 'g0', 4, 'f0', dt.SINGLE],
        ['load_vector_dist1', 'g0', 4, 'v0', dt.DOUBLE],
        ['load_vector_dist1', 'g0', 4, 'v0', dt.SINGLE],
        ['load_vector_dist1_boff', 'g0', 4, 'v0', dt.DOUBLE],
        ['load_vector_dist1_boff', 'g0', 4, 'v0', dt.SINGLE],
        ['load_vector_dist1_inc', 'g0', 4, 'v0', dt.DOUBLE],
        ['load_vector_dist1_inc', 'g0', 4, 'v0', dt.SINGLE],
        ['load_vector_immstride', 'g0', 4, 'v0', dt.SINGLE],
        ['load_vector_gregstride', 'g0', 'g1', 'v0', dt.SINGLE],
        ['load_vector_gather', 'g0', 'v1', 'v0', dt.SINGLE, it.INT32],
        ['store_vector_immstride', 'g0', 4, 'v0', dt.SINGLE],
        ['store_vector_gregstride', 'g0', 'g1', 'v0', dt.SINGLE],
        ['store_vector_scatter', 'g0', 'v1', 'v0', dt.SINGLE, it.INT32],
    ])
    def test_asmgen_method_implemented(self, name, *argv):
        if name in self.allowed:
            self.skipTest(f"Generator allowed to not implement {name}")
        args_list = []
        def regparam_to_reg(param : str) -> Union[greg,freg,vreg,str]:
                m = re.match(r"v(\d)", arg)
                if m:
                    return self.gen.vreg(int(m[1]))
                m = re.match(r"g(\d)", arg)
                if m:
                    return self.gen.greg(int(m[1]))
                m = re.match(r"f(\d)", arg)
                if m:
                    return self.gen.freg(int(m[1]))
                return param
        for arg in argv:
            if type(arg) == str:
                arg = regparam_to_reg(arg)
            args_list.append(arg)
        method = getattr(self.gen, name)
        if callable(method):
            method(*args_list)

def main():
    unittest.main()

if "__main__" == __name__:
    main()
