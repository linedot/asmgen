# ------------------------------------------------------------------------------
# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@fz-juelich.de>
# Copyright (C) 2021 Stepan Nassyr <s.nassyr@xcpp.org>
# ------------------------------------------------------------------------------
"""
Test operand allocation and validation using operand constraints
"""
import unittest
from dataclasses import dataclass

from asmgen.asmblocks.operand.constraint import (
    operand_constraint,
    minmax_constraint,
)

from asmgen.registers import (
    asm_data_type as adt,
    reg_tracker,
    greg_base,
    freg_base,
    vreg_base,
    treg_base
)


class dummy_greg(greg_base):
    def __init__(self, reg_idx : int):
        self.reg_idx = reg_idx

    @property
    def idx(self):
        return self.reg_idx

    def __str__(self):
        return f"g{self.reg_idx}"

class dummy_vreg(vreg_base):
    def __init__(self, reg_idx : int):
        self.reg_idx = reg_idx

    @property
    def idx(self):
        return self.reg_idx

    def __str__(self):
        return f"v{self.reg_idx}"

class dummy_freg(freg_base):
    def __init__(self, reg_idx : int):
        self.reg_idx = reg_idx

    @property
    def idx(self):
        return self.reg_idx

    def __str__(self):
        return f"f{self.reg_idx}"
    
class dummy_treg(treg_base):
    def __init__(self, reg_idx : int):
        self.reg_idx = reg_idx

    @property
    def idx(self):
        return self.reg_idx
    
    def __str__(self):
        return f"t{self.reg_idx}"

@dataclass(kw_only=True)
class vreg_index_constraint(intval_constraint):
    """
    Constraint on register indices for dummy vregs
    """
    what: str ='index'
    getint: Callable[[value_type],int] = lambda v : v.idx
    makeval: Callable[[int],value_type] = lambda i : dummy_vreg(reg_idx=i)

@dataclass(kw_only=True)
class freg_index_constraint(intval_constraint):
    """
    Constraint on register indices for dummy fregs
    """
    what: str ='index'
    getint: Callable[[value_type],int] = lambda v : v.idx
    makeval: Callable[[int],value_type] = lambda i : dummy_freg(reg_idx=i)

@dataclass(kw_only=True)
class vreg_constraint(vreg_index_constraint):
    """
    Constraint for generating dummy vregs
    """
    def validate(self, name, val, context, params):
        pass

    def valid_values(self, name, context, params):
        for i in range(32):
            yield dummy_vreg(reg_idx=i)

@dataclass(kw_only=True)
class freg_constraint(freg_index_constraint):
    """
    Constraint for generating dummy fregs
    """
    def validate(self, name, val, context, params):
        pass

    def valid_values(self, name, context, params):
        for i in range(32):
            yield dummy_freg(reg_idx=i)

class test_operand_constraints(unittest.TestCase):
    """
    Tests for operand constraints
    """

    def setUp(self):
        self.rt = reg_tracker(reg_type_init_list=[
            ('greg',16),
            ('freg',16),
            ('vreg',16),
            ('treg',16),
        ])

        # 
        [vreg_constraint(), freg_constraint()]

    def test_allocate_vregs(self):
        pass

    def test_allocate_vfv(self):
        pass

    def test_allocate_oneof(self):
        pass
