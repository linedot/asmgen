"""
AVX fmul instruction
"""

from ..operations import modifier

from .avx_opd3_base import avx_opd3_base

class avx_fmul(avx_opd3_base):
    """
    AVX implementation of fmul
    """

    def get_base_inst(self, modifiers):
        return "vmul"

    def check_modifiers(self, modifiers : set[modifier]):
        super().check_modifiers(modifiers=modifiers)
        if modifier.NP in modifiers:
            raise ValueError("AVX fmul has no NP form")
