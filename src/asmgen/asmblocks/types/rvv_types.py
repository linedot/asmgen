"""
RVV 1.0 and 0.7.1 register types
"""
from ...registers import vreg_base

#pylint: disable=too-few-public-methods
class rvv_vreg(vreg_base):
    """
    RVV vector register
    """
    def __init__(self, reg_idx : int):
        self.reg_str = f"v{reg_idx}"

    def __str__(self) -> str:
        return self.reg_str
