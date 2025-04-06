"""
RVV 1.0 and 0.7.1 multiplication
"""

from ..operations import modifier

from .rvv_opd3_base import rvv_opd3_base

class rvv_fmul(rvv_opd3_base):
    """
    RVV 1.0 and 0.7.1 implementation of fma
    """

    def get_base_inst(self, modifiers : set[modifier]):
        return "mul"
