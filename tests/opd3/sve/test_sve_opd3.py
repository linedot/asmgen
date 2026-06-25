# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Contains the base class for SVE opd3 tests
"""
import unittest

from asmgen.asmblocks.sve import sve

class test_sve_opd3(unittest.TestCase):
    """
    Tests SVE opd3 operations
    """

    def setUp(self):
        """
        Sets up the generators for all tests
        """
        self.gen = sve()
        self.gen.set_output_inline(yesno=False)
