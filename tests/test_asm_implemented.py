# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Test the asmgen implementations for different ISAs on whether they implement
required methods
"""
import unittest
import inspect

from parameterized import parameterized, parameterized_class

from asmgen.asmblocks.noarch import asmgen
from asmgen.registers import asm_data_type as adt
from asmgen.registers import asm_index_type as ait

from asmgen.asmblocks.avx_fma import fma128,fma256,avx512
from asmgen.asmblocks.neon import neon
from asmgen.asmblocks.sve import sve
from asmgen.asmblocks.rvv import rvv
from asmgen.asmblocks.rvv071 import rvv071

from .testcase import testcase


allowed_not_implemented = {
        'rvv' : ['load_vector_dist1_inc',
                 'load_vector_dist1_boff',
                 'load_vector_voff',
                 'load_vector_immstride',
                 'store_vector_voff',
                 'store_vector_immstride',
                 'add_greg_voff',
                 'fma_idx',
                 'fma_np_idx',
                 ],
        'rvv071' : ['load_vector_dist1_inc',
                    'load_vector_dist1_boff',
                    'load_vector_voff',
                    'load_vector_immstride',
                    'store_vector_voff',
                    'store_vector_immstride',
                    'add_greg_voff',
                    'fma_idx',
                    'fma_np_idx',
                    ],
        'sve' : ['load_vector_dist1_inc',
                 'load_vector_immstride',
                 'load_vector_gregstride',
                 'store_vector_immstride',
                 'store_vector_gregstride',
                 'fma_vf',
                 'fma_np_vf',
                 ],
        'neon' : ['load_vector_dist1_boff',
                  'load_vector_immstride',
                  'load_vector_gregstride',
                  'load_vector_gather',
                  'store_vector_immstride',
                  'store_vector_gregstride',
                  'store_vector_scatter',
                  'fma_vf',
                  'fma_np_vf',
                  ],
        'fma128' : ['load_vector_dist1_inc',
                    'load_vector_immstride',
                    'load_vector_gregstride',
                    'store_vector_immstride',
                    'store_vector_gregstride',
                    'store_vector_scatter',
                    'fma_idx',
                    'fma_np_idx',
                    'fma_vf',
                    'fma_np_vf',
                    ],
        'fma256' : ['load_vector_dist1_inc',
                    'load_vector_immstride',
                    'load_vector_gregstride',
                    'store_vector_immstride',
                    'store_vector_gregstride',
                    'store_vector_scatter',
                    'fma_idx',
                    'fma_np_idx',
                    'fma_vf',
                    'fma_np_vf',
                    ],
        'avx512' : ['load_vector_dist1_inc',
                    'load_vector_immstride',
                    'load_vector_gregstride',
                    'store_vector_immstride',
                    'store_vector_gregstride',
                    'fma_idx',
                    'fma_np_idx',
                    'fma_vf',
                    'fma_np_vf',
                    ]
}


function_tests = [
        ['simd_size', None],
        ['c_simd_size_function', None],
        ['is_vla', None],
        ['indexable_elements', {'dt' : lambda gen : adt.DOUBLE} ],
        ['max_vregs', None],
        ['max_gregs', None],
        ['max_fregs', None],
        ['min_prefetch_offset', None],
        ['max_prefetch_offset', None],
        ['min_load_voff', None],
        ['max_load_voff', None],
        ['min_load_immoff', {'dt' : lambda gen : adt.DOUBLE} ],
        ['max_load_immoff', {'dt' : lambda gen : adt.DOUBLE} ],
        ['min_load_immoff', {'dt' : lambda gen : adt.SINGLE} ],
        ['max_load_immoff', {'dt' : lambda gen : adt.SINGLE} ],
        ['freg', { 'reg_idx' : lambda gen :  0, 'dt' : lambda gen : adt.DOUBLE} ],
        ['greg', { 'reg_idx' : lambda gen :  0} ],
        ['vreg', { 'reg_idx' : lambda gen :  0} ],
        # We start doing a little trick, because we can't use self here:
        # g0 -> self.gen.greg(0), v0 -> self.gen.vreg(0), f0 -> self.gen.freg(0)
        ['loopbegin', {'reg' : lambda gen: gen.greg(0),
                       'label' : lambda gen : 'someloop'}],
        ['loopbegin_nz', {'reg' : lambda gen: gen.greg(0),
                         'label' : lambda gen : 'someloop',
                         'labelskip' : lambda gen : 'someloop_nz'}],
        ['loopend', {'reg' : lambda gen: gen.greg(0),
                     'label' : lambda gen : 'someloop'}],
        ['zero_greg', {'greg' : lambda gen : gen.greg(0)}],
        ['mov_greg', {'src' : lambda gen : gen.greg(0),
                      'dst' : lambda gen : gen.greg(1)}],
        ['mov_greg_to_param', {'src' : lambda gen : gen.greg(0),
                               'param' : lambda gen : 'someparam'}],
        ['mov_param_to_greg', {'dst' : lambda gen : gen.greg(0),
                               'param' : lambda gen : 'someparam'}],
        ['mov_param_to_greg_shift', {'dst' : lambda gen : gen.greg(0),
                                     'param' : lambda gen : 'someparam',
                                     'bit_count' : lambda gen : 2}],
        ['mov_greg_imm', {'reg' : lambda gen : gen.greg(0),
                          'imm' : lambda gen : 40}],
        ['add_greg_imm', {'reg' : lambda gen : gen.greg(0),
                          'imm' : lambda gen : 40}],
        ['add_greg_greg', {'dst' : lambda gen : gen.greg(0),
                           'reg1' : lambda gen : gen.greg(1),
                           'reg2' : lambda gen : gen.greg(2)}],
        ['sub_greg_greg', {'dst' : lambda gen : gen.greg(0),
                           'reg1' : lambda gen : gen.greg(1),
                           'reg2' : lambda gen : gen.greg(2)}],
        ['add_greg_voff', {'reg' : lambda gen : gen.greg(0),
                           'offset' : lambda gen : 40,
                           'dt' : lambda gen : adt.DOUBLE}],
        ['add_greg_voff', {'reg' : lambda gen : gen.greg(0),
                           'offset' : lambda gen : 40,
                           'dt' : lambda gen : adt.SINGLE}],
        ['mul_greg_imm', {'dst' : lambda gen : gen.greg(0),
                          'src' : lambda gen : gen.greg(1),
                          'factor' : lambda gen : 40}],
        ['shift_greg_left', {'reg' : lambda gen : gen.greg(0),
                             'bit_count' : lambda gen : 4}],
        ['shift_greg_right', {'reg' : lambda gen : gen.greg(0),
                             'bit_count' : lambda gen : 4}],
        ['zero_vreg', {'vreg' : lambda gen : gen.vreg(0),
                       'dt' : lambda gen : adt.DOUBLE}],
        ['zero_vreg', {'vreg' : lambda gen : gen.vreg(0),
                       'dt' : lambda gen : adt.SINGLE}],
        ['prefetch_l1_boff', {'areg'   : lambda gen : gen.greg(0),
                              'offset' : lambda gen : 256}],
        ['load_pointer', {'areg' : lambda gen : gen.greg(0),
                          'name' : lambda gen : 'someparam'}],
        ['load_vector_voff', {'areg' : lambda gen : gen.greg(0),
                              'voffset' : lambda gen : 4,
                              'vreg' : lambda gen : gen.vreg(0),
                              'dt' :  lambda gen : adt.DOUBLE}],
        ['load_vector_voff', {'areg' : lambda gen : gen.greg(0),
                              'voffset' : lambda gen : 4,
                              'vreg' : lambda gen : gen.vreg(0),
                              'dt' :  lambda gen : adt.SINGLE}],
        ['load_scalar_immoff', {'areg' : lambda gen : gen.greg(0),
                                'offset' : lambda gen : 4,
                                'freg' : lambda gen : gen.freg(0, adt.DOUBLE),
                                'dt' :  lambda gen : adt.DOUBLE}],
        ['load_vector_dist1', {'areg' : lambda gen : gen.greg(0),
                               'vreg' : lambda gen : gen.vreg(0),
                               'dt' :  lambda gen : adt.DOUBLE}],
        ['load_vector_dist1', {'areg' : lambda gen : gen.greg(0),
                               'vreg' : lambda gen : gen.vreg(0),
                               'dt' :  lambda gen : adt.SINGLE}],
        ['load_vector_dist1_boff', {'areg' : lambda gen : gen.greg(0),
                                    'offset' : lambda gen : 8,
                                    'vreg' : lambda gen : gen.vreg(0),
                                    'dt' :  lambda gen : adt.DOUBLE}],
        ['load_vector_dist1_boff', {'areg' : lambda gen : gen.greg(0),
                                    'offset' : lambda gen : 4,
                                    'vreg' : lambda gen : gen.vreg(0),
                                    'dt' :  lambda gen : adt.SINGLE}],
        ['load_vector_dist1_inc', {'areg' : lambda gen : gen.greg(0),
                                   'offset' : lambda gen : 8,
                                   'vreg' : lambda gen : gen.vreg(0),
                                   'dt' :  lambda gen : adt.DOUBLE}],
        ['load_vector_dist1_inc', {'areg' : lambda gen : gen.greg(0),
                                   'offset' : lambda gen : 4,
                                   'vreg' : lambda gen : gen.vreg(0),
                                   'dt' :  lambda gen : adt.SINGLE}],
        ['load_vector_immstride', {'areg' : lambda gen : gen.greg(0),
                                   'byte_stride' : lambda gen : 4,
                                   'vreg' : lambda gen : gen.vreg(0),
                                   'dt' :  lambda gen : adt.SINGLE}],
        ['load_vector_gregstride', {'areg' : lambda gen : gen.greg(0),
                                    'sreg' : lambda gen : gen.greg(1),
                                    'vreg' : lambda gen : gen.vreg(0),
                                    'dt' :  lambda gen : adt.SINGLE}],
        ['load_vector_gather', {'areg' : lambda gen : gen.greg(0),
                                'offvreg' : lambda gen : gen.vreg(1),
                                'vreg' : lambda gen : gen.vreg(0),
                                'dt' :  lambda gen : adt.SINGLE,
                                'it' :  lambda gen : ait.INT32}],
        ['store_vector_immstride', {'areg' : lambda gen : gen.greg(0),
                                    'byte_stride' : lambda gen : 4,
                                    'vreg' : lambda gen : gen.vreg(0),
                                    'dt' :  lambda gen : adt.SINGLE}],
        ['store_vector_gregstride', {'areg' : lambda gen : gen.greg(0),
                                     'sreg' : lambda gen : gen.greg(1),
                                     'vreg' : lambda gen : gen.vreg(0),
                                     'dt' :  lambda gen : adt.SINGLE}],
        ['store_vector_scatter', {'areg' : lambda gen : gen.greg(0),
                                  'offvreg' : lambda gen : gen.vreg(1),
                                  'vreg' : lambda gen : gen.vreg(0),
                                  'dt' :  lambda gen : adt.SINGLE,
                                  'it' :  lambda gen : ait.INT32}],
    ]

@parameterized_class([
    {"name": "fma128", "gen": fma128(), "allowed": allowed_not_implemented['fma128']},
    {"name": "fma256", "gen": fma256(), "allowed": allowed_not_implemented['fma256']},
    {"name": "avx512", "gen": avx512(), "allowed": allowed_not_implemented['avx512']},
    {"name": "neon", "gen": neon(), "allowed": allowed_not_implemented['neon']},
    {"name": "sve", "gen": sve(), "allowed": allowed_not_implemented['sve']},
    {"name": "rvv", "gen": rvv(), "allowed": allowed_not_implemented['rvv']},
    {"name": "rvv071", "gen": rvv071(), "allowed": allowed_not_implemented['rvv071']},
])
class asm_implementation_test(unittest.TestCase, testcase):
    """
    Parameterized test that tests implementation of multiple methods in multiple generators 
    """
    @parameterized.expand(function_tests)
    def test_asmgen_method_implemented(self, name, args):
        """
        Tests whether a method is implemented in a generator by calling it with
        reasonable arguments, resulting in a fail if the call fails
        """
        if name in self.allowed:
            self.skipTest(f"Generator allowed to not implement {name}")

        copy_args = {}
        if args:
            for k in args.keys():
                copy_args[k] = args[k](self.gen)

        method = getattr(self.gen, name)
        if callable(method):
            method(**copy_args)

class asm_test_implementation_tests(unittest.TestCase):
    """
    Tests the implementation tester for completeness
    """
    def test_all_funcs_tested(self):
        """
        Tests whether all methods that the base asmgen class declares
        are being tested
        """

        allowed_not_tested= [
            '__init__',
            'operands',
            'set_output_inline',
            'asmwrap',
        ]

        existing_methods = inspect.getmembers(asmgen,
                                              predicate=inspect.isfunction)
        existing_methods = [method[0] for method in existing_methods if \
                            method[0] not in allowed_not_tested]
        tested_methods = [method[0] for method in function_tests]

        untested = 0
        for method in existing_methods:
            if method not in tested_methods:
                untested += 1
                print(f"Implementation test for asmgen method {method} missing")
        self.assertEqual(untested, 0)
