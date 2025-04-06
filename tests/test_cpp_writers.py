"""
Test the C++ source code writers
"""

import unittest

from asmgen.cppgen.writers import write_asmblock_func

class cpp_writers_test(unittest.TestCase):
    """
    Test case for C++ source code writers
    """
    def test_asmblock_func_no_tmpl_left(self):
        """
        Tests for no remaining template parameters in generated C++ code
        """
        cppsource = write_asmblock_func("fake_func", "",
                                        {
                                            "function_params":"",
                                            "prepare":"",
                                            })
        self.assertNotIn("${", cppsource)
