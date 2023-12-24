from asmgen.cppgen.writers import write_asmblock_func

import unittest


class cpp_writers_test(unittest.TestCase):
    def test_asmblock_func_no_tmpl_left(self):
        cppsource = write_asmblock_func("fake_func", "", 
                                        {
                                            "function_params":"",
                                            "prepare":"",
                                            })
        self.assertNotIn("${", cppsource)


def main():
    unittest.main()


if "__main__" == __name__:
    main()
